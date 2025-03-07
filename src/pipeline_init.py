import torch
from openood.trainers.lr_scheduler import cosine_annealing
import wandb
import signal
from types import FrameType
from timm.scheduler import CosineLRScheduler


def get_optimizer(
    net,
    config,
):
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=True,
    )

    # elif config.optimizer.name == "adamw":
    #     optimizer = torch.optim.AdamW(
    #         net.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    #     )

    return optimizer


def get_scheduler(optimizer, config, train_loader):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            config.optimizer.num_epochs * len(train_loader),
            1,
            1e-6 / config.optimizer.lr,
        ),
    )

    # steps_per_epoch = int(len(train_loader) / config.dataset.train.batch_size) + 1
    # total_steps = int(config.optimizer.num_epochs * steps_per_epoch)
    # warmup_steps = int(config.scheduler.warmup_epochs * steps_per_epoch)

    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=(total_steps - warmup_steps),
    #     warmup_t=warmup_steps,
    #     warmup_prefix=True,
    #     cycle_limit=1,
    #     t_in_epochs=False,
    # )

    return scheduler


def signal_handler(signum: int, frame: FrameType | None):
    """Called before the job gets pre-empted or reaches the time-limit.

    This should run quickly. Performing a full checkpoint here mid-epoch is not recommended.
    """
    signal_enum = signal.Signals(signum)
    print(f"Job received a {signal_enum.name} signal!", flush=True)
    # logger.error(f"Job received a {signal_enum.name} signal!")
    # Perform quick actions that will help the job resume later.
    # If you use Weights & Biases: https://docs.wandb.ai/guides/runs/resuming#preemptible-sweeps
    if wandb.run:
        wandb.mark_preempting()


def print_metrics(metrics):
    return " - ".join([f"{k}: {v}" for k, v in metrics.items()])
