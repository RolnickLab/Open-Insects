import torch
import argparse
import yaml
import time
import wandb
from src.OpenOOD.openood.networks import ResNet50
from src.OpenOOD.openood.networks import get_network
from src.evaluator import BioEvaluator
from src.datasets.dataloader import get_id_ood_dataloader_folder


def eval_osr_posthoc(num_classes, id_name, checkpoint_path, pp_name, config_path):
    net = ResNet50(num_classes=num_classes)  # update this with get_network?
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    weights = checkpoint["model_state"]
    net.load_state_dict(weights)
    net.eval()
    net.cuda()

    dataloader_dict = get_id_ood_dataloader_folder(config_path)

    with open(config_path) as file:
        config = yaml.safe_load(file)

    evaluator = BioEvaluator(
        net,
        id_name=id_name,
        data_config_path=config_path,
        config_root="/home/mila/y/yuyan.chen/projects/BioOSR_training/configs",
        dataloader_dict=dataloader_dict,
        postprocessor_name=pp_name,
    )

    return evaluator.eval_ood()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BioOOD")
    parser.add_argument("--pp_name")
    parser.add_argument("--slurm_jobid")
    parser.add_argument("--id_name")
    parser.add_argument("--config_path")
    args = parser.parse_args()

    if args.config_path is None:
        args.config_path = f"configs/datasets/ami/posthoc/{args.id_name}.yaml"

    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    # wandb.init(
    #     project=config["wandb_project"],
    #     entity=config["wandb_entity"],
    #     name=f"{args.id_name}_{args.pp_name}",
    #     resume="allow",
    #     config=config,
    # )

    num_classes_dict = {"ne-america": 2497, "w-europe": 2603, "c-america": 636}

    checkpoint_path = f"/network/scratch/y/yuyan.chen/ood_benchmark/weights/baseline/resnet50/{args.slurm_jobid}/checkpoints/model_best.pth"
    print(args.pp_name)
    start_time = time.time()
    result = eval_osr_posthoc(
        num_classes_dict[args.id_name], args.id_name, checkpoint_path, args.pp_name, args.config_path
    )
    print("--- %s seconds ---" % (time.time() - start_time))

    # wandb.log({"result": result, "time": time.time() - start_time})
