import torch
import wandb
import os
from pathlib import Path
import signal
from types import FrameType
from src.OpenOOD.openood.preprocessors import get_preprocessor
from src.OpenOOD.openood.postprocessors import get_postprocessor
from src.OpenOOD.openood.datasets import get_dataloader, get_ood_dataloader
from src.OpenOOD.openood.utils.config import setup_config
from src.OpenOOD.openood.networks import get_network
from src.evaluator import BioEvaluator
import pathlib

# init
config = setup_config()
dataset_name = config.dataset.name

device = "cuda" if torch.cuda.is_available() else "cpu"


wandb.init(
    project=config.wandb.project,
    entity=config.wandb.entity,
    # name=f"{config.dataset.name}_{config.trainer.name}_{config.postprocessor.name}_{config.network.slurm_id}",
    name=f"{config.dataset.name}_{config.trainer.name}_{config.wandb.name}",
    resume="allow",
    config=config,
)


# load weights from checkpoint
# get dataloaders
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
loader_dict = {"id": id_loader_dict, "ood": ood_loader_dict}
net = get_network(config.network)
checkpoint = torch.load(
    config.network.checkpoint,
    weights_only=False,
)
weights = checkpoint["model_state"]

if config.postprocessor.name == "binary":
    keys_to_change = []
    # change name
    for key in weights:
        if "fc" in key or "mlp" in key:
            keys_to_change.append(key)

    for old_key in keys_to_change:
        if "fc" in old_key:
            new_key = "backbone." + old_key
            weights[new_key] = weights.pop(old_key)
        elif "mlp" in old_key:

            new_key = old_key.replace("mlp", "proj_head")
            print(new_key)
            weights[new_key] = weights.pop(old_key)


net.load_state_dict(weights)
net.eval()
net.cuda()
# postprocessor = get_postprocessor(config)

save_arrays = True
# if config.ood_dataset.name == "ami_bci":
#     save_arrays = False
evaluator = BioEvaluator(
    net,
    config=config,
    dataloader_dict=loader_dict,
    postprocessor_name=config.postprocessor.name,
    save_arrays=save_arrays,
)

results = evaluator.eval_ood()


dataset_name = config.dataset.name
if "ami_" in config.dataset.name:
    dataset_name = dataset_name[4:]

    # save_dir = f"output/{dataset_name}/{config.network.slurm_id}/{config.trainer.name}"
    # save_dir = f"output/{dataset_name}/{config.trainer.name}"
save_dir = f"output/{dataset_name}/{config.wandb.name}"
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
results.to_csv(f"{save_dir}/{config.postprocessor.name}.csv", index=False)
wandb.log({"result": results})
