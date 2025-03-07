# create subsets
import pandas as pd
import json

dir = "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/"
fn = "03_c-america_test_id.csv"
df = pd.read_csv(dir + fn)
species = [
    "Apela divisa",
    "Arrhenophanes perspicilla",
    "Calledema argenta",
    "Calledema marmorea",
    "Canodia difformis",
    "Ceroctena amynta",
    "Dialithis gemmifera",
    "Euclea buscki",
    "Hemiceras flava",
    "Hemiceras losa",
    "Homoeocera stictosoma",
    "Iscadia purpurascens",
    "Leucopleura ciarana",
    "Lirimiris lignitecta",
    "Magava multilinea",
    "Mesoscia dumilla",
    "Micrathetis tecnion",
    "Nystalea superciliosa",
    "Oxidercia thaumantis",
    "Phobetron hipparchia",
    "Poliopastea auripes",
    "Rhapigia aymara",
    "Rhuda difficilis",
    "Robinsonia sanea",
    "Semyra bella",
    "Stauropides persimilis",
    "Truncaptera guatemalensis",
]

df = df[df["species"].isin(species)]

# df = df.groupby("species1", group_keys=False).apply(lambda group: group.sample(n=min(len(group), 2), random_state=42))
# df.to_csv(f"data/{fn}", index=False)

# ood_dir = "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/"


# def _convert_to_txt(df, fn, category_map=None, is_ood=False):
#     df = df.fillna(0)
#     df = df.astype({"speciesKey": "int64"})
#     if is_ood:
#         labels = [-1 for _ in range(len(df))]
#     else:
#         labels = [category_map[str(s)] for s in df.speciesKey]

#     lines = [f"{image_path} {label}" for (image_path, label) in zip(df.image_path, labels)]

#     with open("data/" + fn, "w") as f:
#         for line in lines:
#             f.write(f"{line}\n")


# df = pd.read_csv(
#     "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/aux_labeled/01_ne-america_train_aux_merged_20.csv"
# )
# df = df.groupby("speciesKey", group_keys=False).apply(lambda group: group.sample(n=min(len(group), 1), random_state=42))
# lines = [f"{image_path} {label}" for (image_path, label) in zip(df.image_path, df.speciesKey)]
# with open("./data/01_ne-america_train_aux_merged.txt", "w") as f:
#     for line in lines:
#         f.write(f"{line}\n")


# filenames = [
#     "01_ne-america_test_ood_local.csv",
#     "01_ne-america_train_id.csv",
#     "01_ne-america_test_id.csv",
#     "01_ne-america_val_id.csv",
#     "01_ne-america_val_ood.csv",
#     "01_ne-america_val_ood_local.csv",
# ]

# with open(
#     "/network/scratch/y/yuyan.chen/ood_benchmark/ami/metadata/csv/01_ami-gbif_fine-grained_ne-america_category_map.json",
#     "r",
# ) as file:
#     category_map = json.load(file)

# for fn in filenames:
#     df = pd.read_csv(ood_dir + fn)
#     df = df.groupby("species", group_keys=False).apply(
#         lambda group: group.sample(n=min(len(group), 3), random_state=42)
#     )
#     df.to_csv(f"data/{fn}", index=False)

#     is_ood = "id" not in fn
#     _convert_to_txt(df, fn.split(".")[0] + ".txt", category_map=category_map, is_ood=is_ood)


# fn = "05_test_ood_non-local.csv"
# df = pd.read_csv(ood_dir + fn)
# df = df.groupby("species", group_keys=False).apply(lambda group: group.sample(n=min(len(group), 1), random_state=42))
# df.to_csv(f"data/{fn}", index=False)
# is_ood = "id" not in fn
# _convert_to_txt(df, fn.split(".")[0] + ".txt", category_map=category_map, is_ood=is_ood)

# print(len(df))
# fn = "06_test_ood_non-moth.csv"
# df = pd.read_csv(ood_dir + fn)
# df = df.sample(n=100, random_state=42)
# df.to_csv(f"data/{fn}", index=False)
# is_ood = "id" not in fn
# _convert_to_txt(df, fn.split(".")[0] + ".txt", category_map=category_map, is_ood=is_ood)

# print(len(df))
