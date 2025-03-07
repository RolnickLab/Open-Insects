import numpy as np
import pandas as pd
from scipy import stats


def get_species(df, pred, score):
    true_labels = df["speciesKey"].to_numpy()
    species = df["speciesKey"].unique()
    severity_list = []
    species_list = []
    pred_list = []
    for s in species:
        pred_s = pred[true_labels == s]
        if len(pred_s) > 30:
            softmax_s = score[true_labels == s]
            severity_list.append(softmax_s.mean())
            species_list.append(s)
            mode, count = stats.mode(pred_s)
            pred_list.append(mode)

    severity_list = np.array(severity_list)
    species_list = np.array(species_list)
    pred_list = np.array(pred_list)
    percentiles = [10, 30, 50, 70, 90]
    percentile_values = np.percentile(severity_list, percentiles)
    indices = np.array([np.abs(severity_list - value).argmin() for value in percentile_values])
    return species_list[indices], pred_list[indices]


# save_dirs = [
#     f"/network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/{job_id}/checkpoints"
#     for job_id in [5749459, 5749458, 5594655]
# ]

save_dirs = [
    "/network/scratch/y/yuyan.chen/ood_benchmark/weights/energy/ami/ne-america/5817773/checkpoints/ami_ne-america/energy/mls",
    "/network/scratch/y/yuyan.chen/ood_benchmark/ami/classifier/resnet50/5749458/checkpoints/ami_w-europe/base/temperature_scaling/",
    "/network/scratch/y/yuyan.chen/ood_benchmark/weights/energy/ami/c-america/5914382/checkpoints/ami_c-america/energy/ebo"
    ]

dfs = [
    pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/01_ne-america_test_ood_local.csv"),
    pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/02_w-europe_test_ood_local.csv"),
    pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/03_c-america_test_ood_local.csv"),
]

for save_dir, df in zip(save_dirs, dfs):
    pred = np.load(f"{save_dir}/nearood_local_pred.npy")
    conf = np.load(f"{save_dir}/nearood_local_conf.npy")
    # df = pd.read_csv("/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/05_test_ood_non-local.csv")
    # pred = np.load(f"{save_dir}/farood_non-local_logits.npy")
    # softmax = np.load(f"{save_dir}/farood_non-local_softmax.npy")

    print(get_species(df, pred, conf))



# NE-America Energy MLS
# W-Europe TempScale
# C-America Energy
# C-America NovelBranch