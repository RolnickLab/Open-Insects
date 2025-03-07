import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing
from timm.scheduler import CosineLRScheduler
from pandas import DataFrame
from src.datasets.dataloader import get_osr_loader

device = "cuda" if torch.cuda.is_available() else "cpu"


class ContrastiveTrainer:
    def __init__(self, net: nn.Module, train_df: DataFrame, config: Config) -> None:

        self.net = net
        self.train_df = train_df
        self.config = config

        self.optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"],
        )

        steps_per_epoch = int(config.dataset.train.len / config.dataset.train.batch_size) + 1
        total_steps = int(config.optimizer.num_epochs * steps_per_epoch)
        warmup_steps = int(config.optimizer.warmup_epochs * steps_per_epoch)

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=(total_steps - warmup_steps),
            warmup_t=warmup_steps,
            warmup_prefix=True,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def train_epoch(self, epoch_idx):
        num_classes = self.config.dataset.num_classes
        self.net.train()
        n = 1 if epoch_idx == 0 else 2
        train_loader, train_df = get_osr_loader(self.config, self.train_df, n)
        loss_avg = 0.0
        train_dataiter = iter(train_loader)
        total_steps = (int(self.config.dataset.train.len / self.config.dataset.train.batch_size) + 1) * epoch_idx

        all_embeddings_cur = torch.tensor([], dtype=torch.float32, device=device)
        all_labels_cur = torch.tensor([], dtype=torch.float32, device=device)
        y_ref_all = torch.tensor([], device=device)
        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            images1 = batch["data"].cuda()
            labels1 = batch["label"].cuda()

            logits, features1 = self.net(images1, return_feat_post=True)

            labels2 = get_ref_label(logits, labels1, self.config.dataset.num_classes)
            y_ref_all = torch.cat([y_ref_all, labels2], axis=0)

            known_embedding, known_label = (
                features1[labels1 != -1],
                labels1[labels1 != -1],
            )

            # update embedding
            embed_count_per_species = torch.bincount(all_labels_cur.int(), minlength=num_classes).to(device)

            labels_to_include = torch.arange(num_classes, device=device)[
                embed_count_per_species < self.config.dataset.max_count
            ]
            mask = torch.isin(known_label, labels_to_include).to(device)

            all_embeddings_cur = torch.cat((all_embeddings_cur, known_embedding[mask].detach()), axis=0)
            all_labels_cur = torch.cat((all_labels_cur, known_label[mask]), axis=0)

            if epoch_idx == 0:
                continue

            images2 = batch["data"].cuda()
            _, features2 = self.net(images2, return_feat_post=True)

            criterion = DistanceLoss()
            loss_cls = F.cross_entropy(logits[labels1 < num_classes], labels1[labels1 < num_classes])
            # loss_osr = criterion(features1, features2, (labels1 == labels2).float())
            loss_osr = criterion(features1, features2, get_taxnomic_distance(batch))
            loss = 0.5 * (loss_cls + loss_osr)

            # backward
            total_steps += 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(num_updates=total_steps)

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # TODO: change y_pred_all to create balanced pos & neg pairs
        df = get_dataframe(self.train_df, y_ref_all.cpu().numpy(), self.config.dataset.num_classes)  # update df

        self.train_df = df
        self.all_embeddings = all_embeddings_cur
        self.all_labels = all_labels_cur

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced


def get_ref_label(logits, y_true, num_classes):
    _, preds = torch.topk(logits, 2, dim=1)
    top1 = preds[:, 0]
    top2 = preds[:, 1]

    # create positive pairs for known
    y_ref = y_true.clone()
    # create negative pairs for unknown
    y_ref[y_true >= num_classes] = top1[y_true >= num_classes].clone()

    # compute the number of known negative pairs
    known_positions = torch.where(y_true != -1)[0]
    unknown_count = len(y_true[y_true == -1])
    known_count = len(y_true) - unknown_count
    known_neg_count = min(((int(len(y_true) / 2)) - unknown_count), known_count)

    # create negative pairs for known
    wrong_pred = top1.clone()
    wrong_pred[wrong_pred == y_true] = top2[wrong_pred == y_true].clone()
    y_ref[known_positions[:known_neg_count]] = wrong_pred[known_positions[:known_neg_count]]
    return y_ref


def species_count_map(df, num_classes):
    image_count = df.groupby("label1").count()
    image_count = dict(zip(image_count.index, image_count.image1))
    # print(image_count)

    first_index = dict()
    index = 0
    for i in range(num_classes):
        first_index[i] = index
        index += image_count[i]
    return image_count, first_index


def get_dataframe(df, y_pred, num_classes):
    df = df[["species1", "speciesKey1", "genus1", "family1", "label1", "image1"]]
    species_index, image_count = species_count_map(df, num_classes)
    ref_df = get_reference_images(df, species_index, image_count, y_pred, num_classes)
    ref_df = ref_df.rename(columns=lambda col: f"{col[:-1]}2")  # rename 1 to 2

    df.reset_index(inplace=True, drop=True)
    ref_df.reset_index(inplace=True, drop=True)

    return pd.concat([df, ref_df], axis=1)


def get_reference_images(df, species_index, image_count, labels, num_classes):
    """
    sorted_df must be sorted by 'label1'
    """
    rng = np.random.default_rng(42)
    sorted_df = sort_df(df, num_classes)
    indices = []
    for i in labels:
        try:
            i = i.item()
        except:
            i = i
        random_index = rng.integers(low=0, high=image_count[i], size=1)[0]
        off_set = species_index[i]
        indices.append(random_index + off_set)
    return sorted_df.iloc[indices, :]


def sort_df(df, num_classes):
    df = df[df["label1"] < num_classes]
    df1 = df.sort_values(by="label1")

    return df1


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, output1, output2, true_distance):
        distance = F.pairwise_distance(output1, output2)
        criterion = nn.MSELoss()
        return criterion(distance, true_distance)


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        sigmoid = F.sigmoid(distance)
        criterion = nn.BCELoss()
        return criterion(sigmoid, label)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


def get_taxnomic_distance(batch):
    species1, genus1, family1 = batch["species"], batch["genus"], batch["family"]
    species2, genus2, family2 = batch["species_aux"], batch["genus_aux"], batch["family_aux"]
    species1, genus1, family1 = np.array(species1), np.array(genus1), np.array(family1)
    species2, genus2, family2 = np.array(species2), np.array(genus2), np.array(family2)
    distance = np.ones(species1.shape) * 7.0
    distance[(family1 == family2) & (genus1 != genus2)] = 3.0
    distance[(genus1 == genus2) & (species1 != species2)] = 1.0
    distance[species1 == species2] = 0.0

    return torch.from_numpy(distance).float().to(device)
