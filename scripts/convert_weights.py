import torch

checkpoint = torch.load(
    "/network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5749457/checkpoints/model_best.pth",
    weights_only=False,
)
weights = checkpoint["model_state"]
torch.save(weights, "/network/scratch/y/yuyan.chen/ood_benchmark/pretrained_models/model_5749457.pth")
