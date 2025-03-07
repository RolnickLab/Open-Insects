import torch
import wandb
import os
from pathlib import Path
import signal
from types import FrameType
from src.OpenOOD.openood.preprocessors import get_preprocessor
from src.OpenOOD.openood.datasets import get_dataloader
from src.OpenOOD.openood.utils.config import setup_config
from src.OpenOOD.openood.networks import get_network
from src.OpenOOD.openood.trainers import get_trainer
from src.OpenOOD.openood.recorders import get_recorder
from src.OpenOOD.openood.evaluators import get_evaluator
from src.checkpoint_helper import RunState, load_checkpoint, save_checkpoint
from src.pipeline_init import get_scheduler, get_optimizer, signal_handler, print_metrics


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
loader_dict = get_dataloader(config)

if "oe" in loader_dict:
    train_loader = [loader_dict["train"], loader_dict["oe"]]
else:
    train_loader = loader_dict["train"]
val_loader = loader_dict["val"]

# initialize
start_epoch, best_acc = 1, 0.0
if config.trainer.name == "osr":
    start_epoch = 0
net = get_network(config.network)
wandb_run_id = None
run_dir = Path(os.path.join(SCRATCH, config.run_dir))
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = run_dir / SLURM_JOBID / "checkpoints"
checkpoint: RunState | None = load_checkpoint(checkpoint_dir, map_location=device)
trainer = get_trainer(net, train_loader, val_loader, config)
evaluator = get_evaluator(config)

if checkpoint:
    wandb_run_id = checkpoint["wandb_run_id"]
    start_epoch = checkpoint["epoch"] + 1  # +1 to start at the next epoch.
    best_acc = checkpoint["best_acc"]
    trainer.net.load_state_dict(checkpoint["model_state"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    if config.trainer.name != "mos":
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])
    if config.trainer.name == "godin":
        trainer.h_optimizer.load_state_dict(checkpoint["h_optimizer_state"])
        trainer.h_scheduler.load_state_dict(checkpoint["h_scheduler_state"])
    if config.trainer.name == "arpl":
        trainer.criterion.load_checkpoint(checkpoint["criterion"])

    print(
        f"Checkpoints found in {checkpoint_dir}.\nResuming training at epoch {start_epoch} (best_acc={best_acc:.2%}).",
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

# # add checkpointing
for epoch_idx in range(start_epoch, config.optimizer.num_epochs + 1):
    epoch_metrics, epoch_metrics_formated = {}, {}
    net, train_metrics = trainer.train_epoch(epoch_idx)[:2]
    epoch_metrics["train/loss"] = train_metrics["loss"]
    epoch_metrics_formated["train_loss"] = f"{train_metrics['loss']:.4f}"
    test_metrics = evaluator.eval_acc(net, val_loader, epoch_idx=epoch_idx)
    epoch_metrics["val/loss"] = test_metrics["loss"]
    epoch_metrics["val/acc"] = test_metrics["acc"]
    epoch_metrics_formated["val_loss"] = f"{ test_metrics['loss']:.4f}"
    epoch_metrics_formated["val_acc"] = f"{ test_metrics['acc']:.4f}"
    val_accuracy = test_metrics["acc"]
    is_best = val_accuracy > best_acc
    best_acc = max(val_accuracy, best_acc)

    if config.network.name == "arpl_net":
        model_state_dict = net["netF"].state_dict()
    else:
        model_state_dict = net.state_dict()

    optimizer_state = trainer.optimizer.state_dict()
    scheduler_state = trainer.scheduler.state_dict() if config.trainer.name != "mos" else None
    h_optimizer_state = trainer.optimizer.state_dict() if config.trainer.name == "godin" else None
    h_scheduler_state = trainer.optimizer.state_dict() if config.trainer.name == "godin" else None
    criterion = trainer.criterion.state_dict() if config.trainer.name == "arpl" else None
    save_checkpoint(
        checkpoint_dir,
        is_best,
        RunState(
            wandb_run_id=wandb.run.id,
            epoch=epoch_idx + 1,
            model_state=model_state_dict,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            h_optimizer_state=h_optimizer_state,
            h_scheduler_state=h_scheduler_state,
            criterion=criterion,
            best_acc=best_acc,
        ),
    )

    print(f" {print_metrics(epoch_metrics_formated)}", flush=True)
    wandb.log({**epoch_metrics})


print("Finished Training", flush=True)


# save checkpoints seperately for arpl
# load the best model
if config.trainer.name == "arpl":
    model_best = torch.load(checkpoint_dir / "model_best.pth", map_location=device)
    torch.save(model_best["model_state"], checkpoint_dir / "netF.pth")
    torch.save(model_best["criterion"], checkpoint_dir / "criterion.pth")
