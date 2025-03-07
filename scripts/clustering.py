import torch
import wandb
import os
from pathlib import Path
from src.OpenOOD.openood.datasets import get_dataloader, get_ood_dataloader
from src.OpenOOD.openood.utils.config import setup_config
from src.OpenOOD.openood.networks import get_network
from src.clustering.utils import extract_features, clustering, cluster_acc
import pathlib
import numpy as np
import pandas as pd

# init
config = setup_config()
dataset_name = config.dataset.name

device = "cuda" if torch.cuda.is_available() else "cpu"


id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
loader_dict = {"id": id_loader_dict, "ood": ood_loader_dict}
net = get_network(config.network)
checkpoint = torch.load(
    config.network.checkpoint,
    weights_only=False,
)
weights = checkpoint["model_state"]

net.load_state_dict(weights)
net.eval()
net.cuda()

save_dir = ("/").join(config.network.checkpoint.split("/")[:-1])
save_dir += f"/{config.dataset.name}/{config.trainer.name}"
print(save_dir)
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


# # extract features
# features = extract_features(net, ood_loader_dict["nearood"]["local"])
# torch.save(features, f"{save_dir}/local_ood_features.pt")
# features = torch.load(f"{save_dir}/local_ood_features.pt")
# features = features.cpu().numpy()

# # clustering and predict the labels
# y_pred = clustering(config, features)
# np.save(f"{save_dir}/local_ood_pred.npy", y_pred)
y_pred = np.load(f"{save_dir}/local_ood_pred.npy")

## evaluation

## compute ACC
df = pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/03_c-america_test_ood_local.csv")
sorted_taxon = sorted(list(df["speciesKey"].unique()))
taxon_map = {categ: id for id, categ in enumerate(sorted_taxon)}
y_true = np.array([taxon_map[speciesKey] for speciesKey in list(df["speciesKey"])])
ACC = cluster_acc(y_true, y_pred)

print(ACC)
