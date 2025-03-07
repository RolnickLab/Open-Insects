from openood.evaluation_api import Evaluator
from openood.networks import ResNet50
from torchvision.models import ResNet50_Weights
from torch.hub import load_state_dict_from_url
from openood.trainers import get_trainer
from openood.utils.config import setup_config
from evaluator import BioEvaluator
import torch
import yaml
from datasets.dataloader import get_id_ood_dataloader_webdataset, get_id_ood_dataloader_folder
from openood.evaluation_api.postprocessor import postprocessors


def test_dataloader():
    # config_path = "configs/classifier.yaml"
    # dataloader_dict = get_id_ood_dataloader_webdataset(config_path)
    config_path = "configs/classifier_img_folder.yaml"
    dataloader_dict = get_id_ood_dataloader_folder(config_path)

    print(dataloader_dict)

    train_loader = dataloader_dict["ood"]["val"]

    for sample in train_loader:
        print(sample["data"])
        print(sample["label"])
        break

    # print(len(train_loader))
    # train_dataiter = iter(train_loader)

    # print(train_dataiter)
    # print(len(train_dataiter))


def test_bio_evaluator():
    net = ResNet50(num_classes=636)
    checkpoint = torch.load(
        "/network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5594655/checkpoints/model_best.pth",
        weights_only=False,
    )
    weights = checkpoint["model_state"]
    net.load_state_dict(weights)
    net.eval()
    net.cuda()

    # Initialize an evaluator and evaluate with
    # ASH postprocessor

    config_path = "configs/c-america.yaml"

    postprocessor_name_list = list(postprocessors.keys())
    train_wo_data_list = [
        "conf_branch",
        "rotpred",
        "godin",
        "mos",
        "cider",
        "npos",
    ]

    train_w_data_list = [
        "mcd",
    ]
    postprocessor_name_list = [
        # "openmax",  # failed
        # "msp",
        # "temp_scaling",
        # "odin",
        # "mds",
        # "mds_ensemble",
        # "rmds",
        # "gram",
        # "ebo",
        # "opengan",  # failed
        # "react",
        # "mls",
        # "klm",
        # "vim",
        # "knn",
        "dice",
        # "rankfeat",
        # "ash",
        # "she",
        ##  -----
        # "fdbd",
        # "gmm",
        # "patchcore",
        # "gradnorm",
        # "cutpaste",
        # "residual",
        # "ensemble",
        # "dropout",
        # "draem",
        # "dsvdd",
        # "scale",
        # "ssd",  # data_aug
        # "rd4ad",
        # "gen",
        # "nnguide",
        # "relation",
        # "t2fnorm",
        # "reweightood",
    ]

    dataloader_dict = get_id_ood_dataloader_folder(config_path)

    with open(config_path) as file:
        config = yaml.safe_load(file)

    for pp_name in postprocessor_name_list:
        print(f"--- evaluating method {pp_name} ---", flush=True)

        evaluator = BioEvaluator(
            net,
            id_name="c-america",
            data_config_path=config,
            config_root="/home/mila/y/yuyan.chen/projects/BioOOD/configs",
            dataloader_dict=dataloader_dict,
            postprocessor_name=pp_name,
        )

        evaluator.eval_ood()

        # try:
        #     evaluator = BioEvaluator(
        #         net,
        #         id_name="c-america",
        #         config=config,
        #         config_root="/home/mila/y/yuyan.chen/projects/BioOOD/configs",
        #         dataloader_dict=dataloader_dict,
        #         postprocessor_name=pp_name,
        #     )

        #     evaluator.eval_ood()
        # except:
        #     print("failed")


def test_evaluator():
    net = ResNet50()
    weights = ResNet50_Weights.IMAGENET1K_V1
    net.load_state_dict(load_state_dict_from_url(weights.url))
    preprocessor = weights.transforms()
    net.eval()
    net.cuda()

    # Initialize an evaluator and evaluate with
    # ASH postprocessor

    postprocessor_name = "mls"
    print(postprocessor_name)
    evaluator = Evaluator(
        net,
        id_name="cifar10",
        data_root="/network/scratch/y/yuyan.chen/openood/data",
        config_root="/network/scratch/y/yuyan.chen/openood/configs",
        preprocessor=preprocessor,
        postprocessor_name=postprocessor_name,
        batch_size=32,
        num_workers=1,
    )

    # acc = evaluator.eval_acc("id")
    metrics = evaluator.eval_ood()
    print(metrics)


def test_trainer():
    net = ResNet50()
    config_path = "configs/classifier.yaml"
    dataloader_dict = get_id_ood_dataloader_webdataset(config_path)
    train_loader = dataloader_dict["id"]["train"]
    val_loader = dataloader_dict["id"]["val"]
    config = setup_config()
    trainer = get_trainer(net, train_loader, val_loader, config)
    print(trainer)


# test_trainer()

# test_dataloader()
# test_trainer()

test_evaluator()
