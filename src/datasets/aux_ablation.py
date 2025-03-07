import pandas as pd
import json
import numpy as np
from copy import deepcopy
import pathlib


def species_sampler(df, image_count, species_count):
    temp = df[df["image_count"] >= image_count]
    temp = temp.sample(n=species_count, random_state=42)
    return set(temp.speciesKey)


def image_sampler(df, species, n):
    if species is not None:
        df = df[df["speciesKey"].isin(species)]
    df = df.groupby("speciesKey").apply(lambda x: x.sample(n, replace=False, random_state=42)).reset_index(drop=True)
    return df


def rename_columns(df, n):
    df = df.rename(
        columns=lambda col: f"{col}n",
    )

    return df


class AuxDataset:
    def __init__(self, dir_path) -> None:
        self.region_dict = {
            "ne-america": {
                "nelat": 63,
                "nelng": -57,
                "swlat": 42,
                "swlng": -80,
                "index": 1,
                "total_image_count": 160000,
                "num_classes": 2497,
            },
            "w-europe": {
                "nelat": 61,
                "nelng": 16,
                "swlat": 49,
                "swlng": -9,
                "index": 2,
                "total_image_count": 160000,
                "num_classes": 2603,
            },
            "c-america": {
                "nelat": 10,
                "nelng": -77,
                "swlat": 7,
                "swlng": -83,
                "index": 3,
                "total_image_count": 80000,
                "num_classes": 636,
            },
        }
        self.metadata_dir = dir_path
        self.csv_dir = self.metadata_dir + "csv/"
        self.aux_dir = self.metadata_dir + "aux/"
        self.train_dir = self.metadata_dir + "train/"
        self.aux_labeled = self.metadata_dir + "aux_labeled/"

    def get_info(self):
        for region in self.region_dict:
            print(region)
            df_aux = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux.csv")
            print("total images: ", len(df_aux))
            df_aux = df_aux.groupby("speciesKey").size().reset_index(name="image_count")
            print("total species: ", len(df_aux))
            # num_list = [10, 20, 40, 80, 100, 200, 400]
            num_list = [10, 20, 40, 80, 160, 320]

            for num in num_list:
                species_count = len(df_aux[df_aux["image_count"] >= num])
                print(f"number of species with more than {num} images: {species_count} | total: {species_count * num}")

    def sample_species_and_images(self):
        for region in self.region_dict:
            df_train = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv")
            # convert speciesKey to label
            train_species = sorted(list(df_train.speciesKey.unique()))
            categories_map = {categ: id for id, categ in enumerate(train_species)}
            df_train["label"] = [categories_map[s] for s in df_train.speciesKey]
            df_train["image"] = df_train.image_path

            df_aux = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux.csv")

            df_aux["label"] = [-1 for _ in range(len(df_aux))]
            df_aux["image"] = df_aux.image_path

            df_merged = pd.concat(
                [
                    df_train[["speciesKey", "species", "genus", "family", "image", "label"]],
                    df_aux[["speciesKey", "species", "genus", "family", "image", "label"]],
                ]
            )
            df_merged = df_merged.rename(columns=lambda x: f"{x}1")
            df_merged.to_csv(
                self.train_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_merged.csv",
                index=False,
            )

            continue

            df_train = df_train[["speciesKey", "species", "genus", "family", "image_path"]]
            # rename
            df_train = df_train.rename(columns=lambda x: f"{x}1")

            df_image_count = df_aux.groupby("speciesKey").size().reset_index(name="image_count")

            total_image_count = self.region_dict[region]["total_image_count"]
            image_count_list = [20 * 2 ** (4 - i) for i in range(5)]
            specis_count_list = [0] + [total_image_count / image_count for image_count in image_count_list]

            aux_species = set()
            species_dict, image_dict = dict(), dict()
            for i in range(len(image_count_list)):
                df_image_count = df_image_count[
                    ~df_image_count["speciesKey"].isin(aux_species)
                ]  # remove species that have been sampled
                species_count = specis_count_list[i + 1] - specis_count_list[i]

                image_count, species_count = int(image_count_list[i]), int(species_count)

                sampled_species = species_sampler(
                    df_image_count,
                    species_count=species_count,
                    image_count=image_count,
                )

                aux_species |= sampled_species  # update the set of all sampled species

                species_dict[image_count_list[i]] = sampled_species

            for i, img_count in enumerate(image_count_list):

                sampled_images = image_sampler(df_aux, species_dict[img_count], img_count)
                if i == 0:
                    df_aux_sampled = sampled_images
                else:
                    prev_df = image_dict[image_count_list[i - 1]]
                    subset_images = image_sampler(prev_df, None, img_count)
                    df_aux_sampled = pd.concat([sampled_images, subset_images])
                    cur_species = set(df_aux_sampled.speciesKey)
                    prev_species = set(prev_df.speciesKey)
                    assert len(prev_species - cur_species) == 0

                image_dict[img_count] = df_aux_sampled
                pathlib.Path(self.aux_dir).mkdir(parents=True, exist_ok=True)
                df_aux_sampled.to_csv(
                    self.aux_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux_{img_count}.csv"
                )

                df_aux_sampled = df_aux_sampled[["speciesKey", "species", "genus", "family", "image_path"]]
                df_aux_sampled = df_aux_sampled.rename(columns=lambda x: f"{x}1")
                df_aux_sampled["label1"] = [-1 for _ in range(len(df_aux_sampled))]
                df_aux_sampled = pd.concat([df_train, df_aux_sampled])
                df_aux_sampled.to_csv(
                    self.train_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_merged_{img_count}.csv",
                    index=False,
                )

    def to_txt(self, df_train, df_aux, fn):
        df_aux = df_aux.astype({"speciesKey": "int64"})
        aux_species = sorted(list(df_aux.speciesKey.unique()))
        category_map = {str(categ): id + len(df_train.label1.unique()) for id, categ in enumerate(aux_species)}
        labels = [category_map[str(int(s))] for s in df_aux.speciesKey]
        labels = list(df_train.label1) + labels
        image_paths = list(df_train.image1) + list(df_aux.image_path)

        lines = [f"{image_path} {label}" for (image_path, label) in zip(image_paths, labels)]

        with open(self.aux_labeled + f"{fn}.txt", "w") as f:
            for line in lines:
                f.write(f"{line}\n")

        df = pd.DataFrame(list(zip(image_paths, labels)), columns=["image1", "label1"])
        df.to_csv(self.aux_labeled + f"{fn}.csv", index=False)

    def label_aux_set(self):
        for region in self.region_dict:

            df_train = pd.read_csv(self.train_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv")

            for image_count in [20 * 2 ** (4 - i) for i in range(5)]:
                df_aux = pd.read_csv(
                    self.aux_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux_{image_count}.csv"
                )

                self.to_txt(
                    df_train,
                    df_aux,
                    f"{self.region_dict[region]['index']:02d}_{region}_train_aux_merged_{image_count}",
                )

    def txt_for_open_set(self):
        for region in self.region_dict:
            for image_count in [20 * 2 ** (4 - i) for i in range(5)]:
                df_aux = pd.read_csv(
                    self.aux_labeled
                    + f"{self.region_dict[region]['index']:02d}_{region}_train_aux_merged_{image_count}.csv"
                )
                num_classes = self.region_dict[region]["num_classes"]
                open_set = list(df_aux[df_aux["label1"] >= num_classes].label1.unique())
                df_aux["label1"] = df_aux["label1"].replace(open_set, num_classes)

                # convert it to txt

                lines = [f"{image_path} {label}" for (image_path, label) in zip(df_aux.image1, df_aux.label1)]
                fn = f"{self.region_dict[region]['index']:02d}_{region}_train_open_set_{image_count}"
                print(fn)
                with open(self.aux_labeled + f"{fn}.txt", "w") as f:
                    for line in lines:
                        f.write(f"{line}\n")

    def txt_for_oe(self):
        for region in self.region_dict:
            for image_count in [20 * 2 ** (4 - i) for i in range(5)]:
                df_aux = pd.read_csv(
                    self.aux_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux_{image_count}.csv"
                )

                image_paths = df_aux.image1
                labels = [-1 for _ in range(len(image_paths))]

                lines = [f"{image_path} {label}" for (image_path, label) in zip(image_paths, labels)]
                fn = f"{self.region_dict[region]['index']:02d}_{region}_train_oe_{image_count}"
                print(fn)
                with open(self.aux_labeled + f"{fn}.txt", "w") as f:
                    for line in lines:
                        f.write(f"{line}\n")

    def convert_csv(self):
        for region in self.region_dict:
            df_train = pd.read_csv(
                self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv"
            )  # they have species
