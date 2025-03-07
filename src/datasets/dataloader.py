# Copyright 2022 Fagner Cunha
# Copyright 2023 Rolnick Lab at Mila Quebec AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import json
import tarfile

# from absl import flags

import braceexpand
import torch
import webdataset as wds
from torchvision import transforms
import numpy
import random
import pandas as pd
import PIL
from datasets import preprocessing
import os
from pathlib import Path

import yaml
import os


random_seed = 42

random.seed(random_seed)
numpy.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# getting environment variables
SCRATCH = os.environ["SCRATCH"]
try:
    SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
except:
    SLURM_TMPDIR = None
SLURM_JOBID = os.environ["SLURM_JOB_ID"]


def identity(x):
    return x


def get_label_transform(config):
    if config["label_transform_json"] is None:
        return identity
    else:
        with open(config["label_transform_json"], "r") as f:
            label_map = json.load(f)

        label_map = {int(k): int(label_map[k]) for k in label_map}

        return label_map.get


def _load_exclude_list(config):
    with open(config["sample_exclude_list"], "r") as f:
        exclude_list = json.load(f)
    exclude_list = [str(i).lower() for i in exclude_list]
    exclude_list = set(exclude_list)

    return exclude_list


def get_sample_filter_by_key(config):
    exclude_list = _load_exclude_list(config)

    def not_in_exclude_list(sample):
        sample_id = sample["__key__"].split("/")[-1]

        return sample_id not in exclude_list

    return not_in_exclude_list


def geo_prior_preprocess(json_data):
    lat = json_data["decimalLatitude"]
    lon = json_data["decimalLongitude"]
    date = json_data["eventDate"]

    feats, valid = preprocessing.preprocess_loc_date(lat, lon, date, validate=True)
    return feats, valid


def get_dataset(
    config,
    sharedurl,
    input_size,
    is_training,
    preprocess_mode,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    transform = preprocessing.get_image_transforms(config, input_size, is_training, preprocess_mode)
    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=False)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    # if shuffle_samples:
    #     dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        sample = []
        if use_geoprior_data:
            image, cls, json_data, data_key = data
            feats, valid = geo_prior_preprocess(json_data)
            sample += [feats, valid]
        else:
            image, cls, data_key = data

        image = transform(image)
        label = label_transform(cls)

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        sample = [image, label] + sample

        if return_instance_id:
            sample += [data_key]

        return tuple(sample)

    dataset = dataset.map(map_fn)

    return dataset


def get_dataset_simgcd(
    config,
    sharedurl,
    transform,
    is_training,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=False)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    # if shuffle_samples:
    #     dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        sample = []
        if use_geoprior_data:
            image, cls, json_data, data_key = data
            feats, valid = geo_prior_preprocess(json_data)
            sample += [feats, valid]
        else:
            image, cls, data_key = data

        image = transform(image)
        label = label_transform(cls)

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        sample = [image, label] + sample

        if return_instance_id:
            sample += [data_key]

        return tuple(sample)

    dataset = dataset.map(map_fn)

    return dataset


def img_decoder(value):
    with io.BytesIO(value) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("RGB")
    return value, img


def cls_decoder(value):
    return value


def build_webdataset_with_speciesKey(
    config,
    sharedurl,
    input_size,
    batch_size,
    num_workers,
    is_training,
    preprocess_mode,
    load_filename=False,
):
    transform = preprocessing.get_image_transforms(config, input_size, is_training, preprocess_mode)
    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    dataset = dataset.decode("pil").to_tuple("jpg", "cls", "species.cls", "__key__")

    def map_fn(data):
        sample = []
        image, cls, speciesKey, fn = data

        image = transform(image)
        label = label_transform(cls)
        if load_filename:
            sample = [image, label, speciesKey, fn] + sample
        else:
            sample = [image, label, speciesKey] + sample

        return tuple(sample)

    dataset = dataset.map(map_fn)

    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    return loader


class ContrastiveTransformations:
    def __init__(self, base_transforms, aug_transform, n_views=2):
        self.base_transforms = base_transforms
        self.aug_transform = aug_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x), self.aug_transform(x)]


# adapted from SupContrast
class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def build_webdataset_pipeline(
    config,
    sharedurl,
    input_size,
    batch_size,
    num_workers,
    is_training,
    preprocess_mode,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    transform = preprocessing.get_image_transforms(config, input_size, is_training, preprocess_mode)
    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    if shuffle_samples:
        dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        # sample = []
        sample = {}
        if use_geoprior_data:
            image, cls, json_data, data_key = data
            feats, valid = geo_prior_preprocess(json_data)
            # sample += [feats, valid]
            sample["feats"], sample["valid"] = feats, valid
        else:
            image, cls, data_key = data

        image = transform(image)
        label = label_transform(cls)

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        # sample = [image, label] + sample

        # TODO: change preprocessor for data_aux
        sample["data"], sample["data_aux"], sample["label"] = image, image, label

        if return_instance_id:
            # sample += [data_key]
            sample["data_key"] = data_key
        return sample

    dataset = dataset.map(map_fn)
    # TODO: add length
    dataset = dataset.with_length(config["train_data_len"])
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    return loader


def build_webdataset_supcon(config, sharedurl, input_size, batch_size, num_workers, is_training):

    shuffle_samples = is_training

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    dataaug_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), # do not change color
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    original_transform = preprocessing.get_image_transforms(config, input_size, is_training, "torch")

    supcon_transform = TwoCropTransform(dataaug_transform)

    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    if shuffle_samples:
        dataset = dataset.shuffle(10000)

    dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        sample = []
        image, cls, data_key = data
        label = label_transform(cls)
        images = supcon_transform(image)
        sample = [images, label] + sample
        return tuple(sample)

    dataset = dataset.map(map_fn)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    return loader


def build_webdataset_contrastive(
    config,
    sharedurl,
    input_size,
    batch_size,
    num_workers,
    is_training,
    preprocess_mode,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    base_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=input_size),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    id_transform = ContrastiveTransformations(base_transform, base_transform, n_views=2)

    ood_transform = ContrastiveTransformations(base_transform, contrast_transforms, n_views=2)

    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    if shuffle_samples:
        dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "__key__")

    def map_fn(data):
        sample = []
        if use_geoprior_data:
            image, cls, json_data, data_key = data
            feats, valid = geo_prior_preprocess(json_data)
            sample += [feats, valid]
        else:
            image, cls, data_key = data

        label = label_transform(cls)
        if label != -1:
            image1, image2 = id_transform(image)
        else:
            image1, image2 = ood_transform(image)

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        sample = [image1, image2, label] + sample

        if return_instance_id:
            sample += [data_key]
        return tuple(sample)

    dataset = dataset.map(map_fn)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    return loader


def build_trap_dataloader(
    config,
    sharedurl,
    input_size,
    batch_size,
    num_workers,
    is_training,
    preprocess_mode,
    return_instance_id=False,
    use_geoprior_data=False,
    shuffle_samples=None,
    multihead_map=None,
):
    if shuffle_samples is None:
        shuffle_samples = is_training

    transform = preprocessing.get_image_transforms(config, input_size, is_training, preprocess_mode)
    label_transform = get_label_transform(config)

    dataset = wds.WebDataset(sharedurl, shardshuffle=shuffle_samples)

    if config["sample_exclude_list"] is not None:
        dataset = dataset.select(get_sample_filter_by_key(config))

    if shuffle_samples:
        dataset = dataset.shuffle(10000)

    if use_geoprior_data:
        dataset = dataset.decode("pil").to_tuple("jpg", "cls", "json", "__key__")
    else:
        dataset = dataset.decode("pil").to_tuple("jpg", "json")

    def map_fn(data):
        sample = []
        image, annotation = data

        image = transform(image)
        label = annotation["speciesKey"] if annotation["speciesKey"] is not None else -1
        name = annotation["label"] if annotation["label"] is not None else ""

        if multihead_map is not None:
            label = tuple([head[label] for head in multihead_map])

        sample = [image, label, name] + sample

        return tuple(sample)

    dataset = dataset.map(map_fn)

    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    return loader


def _count_files_from_tar(tar_filename, exclude_list=None, ext="jpg"):
    tar = tarfile.open(tar_filename)
    files = [f for f in tar.getmembers() if f.name.endswith(ext)]
    files = [f for f in files if f.name.split("/")[-1][: -(len(ext) + 1)] not in exclude_list]
    count_files = len(files)
    tar.close()
    return count_files


def get_webdataset_length(config, sharedurl):
    if config["sample_exclude_list"] is not None:
        exclude_list = _load_exclude_list(config)
    else:
        exclude_list = {}

    tar_filenames = list(braceexpand.braceexpand(sharedurl))
    counts = [_count_files_from_tar(tar_f, exclude_list) for tar_f in tar_filenames]
    return int(sum(counts))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_folder, transform=None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index].loc["image1"]
        sample = {}
        label = self.df.iloc[index].loc["label1"]
        image = PIL.Image.open(os.path.join(self.images_folder, filename)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        sample["data"], sample["data_aux"], sample["label"] = image, image, label

        return sample


class GCDDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_folder, transform=None, train=False):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index].loc["image1"]
        y_ref = self.df.iloc[index].loc["label1"]

        image = PIL.Image.open(os.path.join(self.images_folder, filename)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.train:
            mask = self.df.iloc[index].loc["mask"]
            return image, y_ref, mask

        return image, y_ref


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_folder, transform=None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = {}
        filename1 = self.df.iloc[index].loc["image1"]
        label1 = self.df.iloc[index].loc["label1"]
        filename2 = self.df.iloc[index].loc["image2"]
        label2 = self.df.iloc[index].loc["label2"]
        species1 = self.df.iloc[index].loc["species1"]
        species2 = self.df.iloc[index].loc["species2"]
        genus1 = self.df.iloc[index].loc["genus1"]
        genus2 = self.df.iloc[index].loc["genus2"]
        family1 = self.df.iloc[index].loc["family1"]
        family2 = self.df.iloc[index].loc["family2"]

        image1 = PIL.Image.open(os.path.join(self.images_folder, filename1)).convert("RGB")
        image2 = PIL.Image.open(os.path.join(self.images_folder, filename2)).convert("RGB")
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        sample["data"], sample["data_aux"], sample["label"], sample["label_aux"] = image1, image2, label1, label2
        sample["species"], sample["genus"], sample["family"] = species1, genus1, family1
        sample["species_aux"], sample["genus_aux"], sample["family_aux"] = species2, genus2, family2

        return sample


def get_osr_loader(
    config,
    df,
    n,
    shuffle=True,
):

    # shuffle dataframe
    if shuffle:
        df = df.sample(frac=1, random_state=config.seed)

    transform = preprocessing.get_image_transforms(config, config.dataset.image_size, False, "torch")

    if n == 1:
        dataset = ImageDataset(df, config.dataset.train.data_dir, transform)
    if n == 2:
        dataset = ImagePairDataset(df, config.dataset.train.data_dir, transform)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.dataset.train.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return loader, df


def get_posthoc_loader(
    config,
    df,
    n,
    shuffle=True,
):

    # shuffle dataframe
    if shuffle:
        df = df.sample(frac=1, random_state=config.seed)

    transform = preprocessing.get_image_transforms(config, config["image_folder"], False, "torch")

    if n == 1:
        dataset = ImageDataset(df, config["image_folder"], transform)
    if n == 2:
        dataset = ImagePairDataset(df, config["image_folder"], transform)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=get_num_workers(),
    )

    return loader, df


def get_num_workers():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


def get_gcd_dataset(config, df, image_folder, shuffle=True, transform=None, is_train=False):

    # shuffle dataframe
    if shuffle:
        df = df.sample(frac=1, random_state=config["random_seed"])

    if transform is None:

        transform = preprocessing.get_image_transforms(
            config, config["input_size"], False, config["preprocessing_mode"]
        )

    dataset = GCDDataset(df, image_folder, transform, is_train)

    return dataset


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def get_id_ood_dataloader_webdataset(config_path):

    def get_num_workers():
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return torch.multiprocessing.cpu_count()

    config = load_config(config_path)
    dataloader_dict = {}
    sharedurl_dict = {"id": {"train", "val", "test"}, "ood": {"val", "near", "far"}}
    for data_type in sharedurl_dict.keys():
        sub_dataloader_dict = {}
        for split in sharedurl_dict[data_type]:
            data_root = os.environ[config["data_root"]]
            url_path = os.path.join(data_root, config[f"{data_type}_{split}"])
            sub_dataloader_dict[split] = build_webdataset_pipeline(
                config,
                sharedurl=url_path,
                input_size=config["input_size"],
                batch_size=config["batch_size"],
                num_workers=get_num_workers(),
                is_training=False,
                preprocess_mode="torch",
                return_instance_id=False,
                use_geoprior_data=False,
            )

        dataloader_dict[data_type] = sub_dataloader_dict

    # TODO: add non-moth!
    ood_near_loader = dataloader_dict["ood"]["near"]
    ood_far_loader = dataloader_dict["ood"]["far"]

    dataloader_dict["ood"]["near"] = {}
    dataloader_dict["ood"]["far"] = {}

    dataloader_dict["ood"]["near"]["local"] = ood_near_loader
    dataloader_dict["ood"]["far"]["non-local"] = ood_far_loader

    return dataloader_dict


def get_id_ood_dataloader_folder(config_path):
    config = load_config(config_path)
    dataloader_dict = {}
    sharedurl_dict = {"id": {"train", "val", "test"}, "ood": {"val", "near", "far"}}
    for data_type in sharedurl_dict.keys():
        sub_dataloader_dict = {}
        for split in sharedurl_dict[data_type]:
            sub_dataloader_dict[split], _ = get_posthoc_loader(
                config, df=pd.read_csv(config[f"{data_type}_{split}"]), n=1, shuffle=False
            )

        dataloader_dict[data_type] = sub_dataloader_dict

    # TODO: add non-moth!
    ood_near_loader = dataloader_dict["ood"]["near"]
    ood_far_loader = dataloader_dict["ood"]["far"]

    dataloader_dict["ood"]["near"] = {}
    dataloader_dict["ood"]["far"] = {}

    dataloader_dict["ood"]["near"]["local"] = ood_near_loader
    dataloader_dict["ood"]["far"]["non-local"] = ood_far_loader
    dataloader_dict["ood"]["far"]["non-moth"], _ = get_posthoc_loader(
        config,
        df=pd.read_csv(config["ood_non_moth"]),
        n=1,
        shuffle=False,
    )

    return dataloader_dict
