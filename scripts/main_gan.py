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
from src.OpenOOD.openood.pipelines import get_pipeline

# init


config = setup_config()
device = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project=config.wandb.project,
    entity=config.wandb.entity,
    name=config.wandb.name,
    resume="allow",
    config=config,
)

pipeline = get_pipeline(config)
pipeline.run()
