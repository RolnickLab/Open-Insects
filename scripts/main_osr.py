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
from src.OpenOOD.openood.trainers import get_trainer
from src.OpenOOD.openood.recorders import get_recorder
from src.OpenOOD.openood.evaluators import get_evaluator
from src.checkpoint_helper import RunState, load_checkpoint, save_checkpoint
from src.pipeline_init import get_scheduler, get_optimizer, signal_handler, print_metrics
from src.datasets.dataloader import get_osr_loader

# init
config = setup_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# getting environment variables
SCRATCH = os.environ["SCRATCH"]
try:
    SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
except:
    SLURM_TMPDIR = None
SLURM_JOBID = os.environ["SLURM_JOB_ID"]
# load weights from checkpoint
# get dataloaders
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
train_dataframe = id_loader_dict["train"]
train_loader, _ = get_osr_loader(config, train_dataframe, 1)
id_loader_dict["train"] = train_loader
val_loader = id_loader_dict["val"]

# initialize
start_epoch, best_auroc = 1, 0.0
if config.trainer.name in ["contrastive", "osr_branch"]:
    start_epoch = 0
net = get_network(config.network)
wandb_run_id = None
run_dir = Path(os.path.join(SCRATCH, config.run_dir))
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = run_dir / SLURM_JOBID / "checkpoints"
checkpoint: RunState | None = load_checkpoint(checkpoint_dir, map_location=device)
trainer = get_trainer(net, train_dataframe, val_loader, config)
evaluator = get_evaluator(config)
postprocessor = get_postprocessor(config)


if checkpoint:
    wandb_run_id = checkpoint["wandb_run_id"]
    start_epoch = checkpoint["epoch"] + 1  # +1 to start at the next epoch.
    best_auroc = checkpoint["best_auroc"]
    trainer.net.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])
    trainer.all_features = checkpoint["all_features"]
    trainer.all_labels = checkpoint["all_labels"]
    trainer.train_df = checkpoint["train_df"]
    print(
        f"Checkpoints found in {checkpoint_dir}.\nResuming training at epoch {start_epoch} (best_acc={best_auroc:.2%}).",
        flush=True,
    )
else:
    print(
        f"No checkpoints found in {checkpoint_dir}. Training from scratch.",
        flush=True,
    )


if wandb_run_id == None:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        resume="allow",
        config=config,
    )
else:
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name + "_" + SLURM_JOBID,
        id=wandb_run_id,
        resume="must",
        config=config,
    )


signal.signal(signal.SIGTERM, signal_handler)  # Before getting pre-empted and requeued.
signal.signal(signal.SIGUSR1, signal_handler)  # Before reaching the end of the time limit.

# add checkpointing
for epoch_idx in range(start_epoch, config.optimizer.num_epochs + 1):
    epoch_metrics, epoch_metrics_formated = {}, {}
    net, train_metrics = trainer.train_epoch(epoch_idx)

    if epoch_idx == 0:
        print(f"train_acc: {train_metrics['acc']:.4f}")
        continue

    epoch_metrics["train/loss"] = train_metrics["loss"]
    epoch_metrics["train/acc"] = train_metrics["acc"]
    epoch_metrics_formated["train_loss"] = f"{epoch_metrics['train/loss']:.4f}"
    epoch_metrics_formated["train_acc"] = f"{epoch_metrics['train/acc']:.4f}"

    postprocessor.all_feats = trainer.all_features
    postprocessor.all_labels = trainer.all_labels
    fpr, auroc, aupr_in, aupr_out, acc = evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
    epoch_metrics["val/acc"] = acc
    epoch_metrics["val/auroc"] = auroc
    epoch_metrics["val/fpr"] = fpr

    epoch_metrics_formated["val_acc"] = f"{ epoch_metrics['val/acc']:.4f}"
    epoch_metrics_formated["val_auroc"] = f"{ epoch_metrics['val/auroc']:.4f}"
    epoch_metrics_formated["val_fpr"] = f"{ epoch_metrics['val/fpr']:.4f}"

    val_acc = acc
    is_best = val_acc > best_auroc
    best_auroc = max(val_acc, best_auroc)

    if config.network.name == "arpl_net":
        if torch.cuda.device_count() > 1:
            model_state_dict = net["netF"].module.state_dict()
        else:
            model_state_dict = net["netF"].state_dict()
    else:

        if torch.cuda.device_count() > 1:
            model_state_dict = net.module.state_dict()
        else:
            model_state_dict = net.state_dict()

    save_checkpoint(
        checkpoint_dir,
        is_best,
        RunState(
            wandb_run_id=wandb.run.id,
            epoch=epoch_idx + 1,
            model_state=model_state_dict,
            optimizer_state=trainer.optimizer.state_dict(),
            scheduler_state=trainer.scheduler.state_dict(),
            best_auroc=best_auroc,
            all_features=trainer.all_features,
            all_labels=trainer.all_labels,
            train_df=trainer.train_df,
        ),
    )

    print(f" {print_metrics(epoch_metrics_formated)}", flush=True)
    wandb.log({**epoch_metrics})


print("Finished Training", flush=True)
