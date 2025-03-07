import pandas as pd
import requests
import json
from functools import partialmethod
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def pairwise_disjoint(sets) -> bool:
    union = set().union(*sets)
    return len(union) == sum(map(len, sets))


# use species not speciesKey
class AMIDataset:
    def __init__(self, ami_dir, ami_binary_dir, meatadata_dir) -> None:
        self.region_dict = {
            "ne-america": {
                "nelat": 63,
                "nelng": -57,
                "swlat": 42,
                "swlng": -80,
                "index": 1,
            },
            "w-europe": {
                "nelat": 61,
                "nelng": 16,
                "swlat": 49,
                "swlng": -9,
                "index": 2,
            },
            "c-america": {
                "nelat": 10,
                "nelng": -77,
                "swlat": 7,
                "swlng": -83,
                "index": 3,
            },
        }

        self.ami_dir = ami_dir
        self.ami_binary_dir = ami_binary_dir
        self.metadata_dir = meatadata_dir
        self.csv_dir = self.metadata_dir + "csv/"
        self.txt_dir = self.metadata_dir + "txt/"
        self.global_moth_all = self.csv_dir + "global_moth_images_filtered.csv"
        self.ami_gbif_all = self.ami_dir + "04_ami-gbif_fine-grained_all_train.csv"

    def create_dataset(self):
        self.extend_taxonomy_map()
        self.match_key_and_name()
        self.create_ood_dataset()
        self.create_aux_dataset()
        self.get_summary()

    def create_world_map(self):
        region_dict = {
            "ne-america": {"nelat": 63, "nelng": -57, "swlat": 42, "swlng": -80, "index": 1},
            "w-europe": {"nelat": 61, "nelng": 16, "swlat": 49, "swlng": -9, "index": 2},
            "c-america": {"nelat": 10, "nelng": -77, "swlat": 7, "swlng": -83, "index": 3},
            "australia": {
                "nelat": -10,
                "nelng": 154,
                "swlat": -44,
                "swlng": 113,
            },
        }

        # Create the plot with a global projection
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_global()
        # Add features to the map
        ax.coastlines()
        # ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, color="lightgray")
        ax.add_feature(cfeature.OCEAN, color="lightblue")
        for region in region_dict.keys():
            min_lat, max_lat = (
                region_dict[region]["swlat"] - 1,
                region_dict[region]["nelat"] + 1,
            )
            min_lon, max_lon = (
                region_dict[region]["swlng"] - 3,
                region_dict[region]["nelng"] + 3,
            )

            # Plot the bounding box using the coordinates
            ax.plot([min_lon, max_lon], [min_lat, min_lat], color="red")  # Bottom edge
            ax.plot([min_lon, max_lon], [max_lat, max_lat], color="red")  # Top edge
            ax.plot([min_lon, min_lon], [min_lat, max_lat], color="red")  # Left edge
            ax.plot([max_lon, max_lon], [min_lat, max_lat], color="red")  # Right edge

        plt.savefig(f"map.png")

    def extend_taxonomy_map(self):
        df_global = pd.read_csv("/network/scratch/y/yuyan.chen/train.csv")
        df_ami = pd.read_csv(self.ami_dir + "04_ami-gbif_fine-grained_all_train.csv")
        species_all = set(df_global.speciesKey) | set(df_ami.speciesKey)
        df_taxonomy_map = pd.read_csv(self.ami_dir + "taxonomy_map.csv")

        species_in_tax_map = set(df_taxonomy_map.speciesKey)
        speciesKey_list = list(species_all - species_in_tax_map)  # 801
        species_list, genus_list, family_list = [], [], []
        speciesKey_list_temp = []
        for i, speciesKey in enumerate(speciesKey_list):
            speciesKey = int(speciesKey)
            species, genus, family = get_taxonomy_name_from_gbif(speciesKey)
            species_list.append(species)
            genus_list.append(genus)
            family_list.append(family)
            speciesKey_list_temp.append(speciesKey)

        df_add = pd.DataFrame(
            list(
                zip(speciesKey_list_temp, species_list, genus_list, family_list),
            ),
            columns=["speciesKey", "species", "genus", "family"],
        )

        df_taxonomy_map = pd.concat([df_taxonomy_map, df_add])
        df_taxonomy_map.to_csv(self.csv_dir + "taxonomy_map.csv", index=False)

    def match_key_and_name(self):
        # merge global dataframe
        # print("Step 2: merge taxonony map and image csv's")
        df_global = pd.read_csv(self.global_moth_all)
        df_ami = pd.read_csv(self.ami_gbif_all)

        self._merge_csv(df_global, "global_moths_with_names.csv")
        self._merge_csv(df_ami, "04_all_id_train.csv")
        for region in self.region_dict.keys():
            data_type_list = ["train", "val", "test"]
            for data_type in data_type_list:
                df = pd.read_csv(
                    self.ami_dir
                    + f"{self.region_dict[region]['index']:02d}_ami-gbif_fine-grained_{region}_{data_type}.csv"
                )
                if region != "c-america" and data_type == "val":
                    # subsample validation set
                    df = df.groupby("speciesKey", group_keys=False).apply(
                        lambda group: group.sample(n=min(len(group), 10), random_state=42)
                    )
                df = self._merge_csv(
                    df,
                    f"{self.region_dict[region]['index']:02d}_{region}_{data_type}_id.csv",
                )

    # step 3
    def create_ood_dataset(self):
        df_val_local_list = []

        for region in self.region_dict.keys():
            _, df_val_local = self._get_local_ood_regional(region)
            df_val_local_list.append(df_val_local)

        _, df_val_nonmoth = self._get_non_moth_ood()
        _, df_val_australia = self._get_non_local_ood()

        for region in self.region_dict.keys():
            df_val = pd.concat([df_val_nonmoth, df_val_australia, df_val_local])
            df_val.to_csv(
                self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_val_ood.csv",
                index=False,
            )

    # step 4
    def create_aux_dataset(self):
        for region in self.region_dict.keys():
            self._create_aux_dataset_regional(region)

    def _create_aux_dataset_regional(self, region):
        # get all moth species
        df_global = pd.read_csv(self.csv_dir + "global_moths_with_names.csv")
        species_all = set(df_global.species)
        # exclude id species
        df_id = pd.read_csv(
            self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv",
        )
        df_ood = pd.read_csv(
            self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_test_ood_local.csv",
        )

        species_id = set(df_id.species)
        species_ood = set(df_ood.species)
        genera_id = set(df_id.genus)
        genera_ood = set(df_ood.genus)

        species_aux = species_all - species_id - species_ood

        # exclude australian species
        df_aus = pd.read_csv(self.csv_dir + "05_test_ood_non-local.csv")
        aus_species = set(df_aus.species)
        aus_genera = set(df_aus.genus)

        species_aux -= aus_species

        df_aux = df_global[df_global["species"].isin(species_aux)]

        # exclude genera that are only in the ood set (both local and non-local)
        local_ood_only_genera = genera_ood - genera_id
        print(local_ood_only_genera)
        print(len(genera_ood & genera_id))
        exit()
        non_local_ood_only_genera = aus_genera - genera_id
        genera_to_exclude = local_ood_only_genera | non_local_ood_only_genera
        df_aux = df_aux[~df_aux["genus"].isin(genera_to_exclude)]

        return

        df_id_val = pd.read_csv(
            self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_val_id.csv",
        )
        df_aux_val = df_aux.sample(n=len(df_id_val), random_state=42)
        df_aux_train = df_aux.drop(index=df_aux_val.index)

        df_aux_val.to_csv(
            self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_val_aux.csv",
            index=False,
        )

        df_aux_train.to_csv(
            self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_aux.csv",
            index=False,
        )
        print("region: ", region)
        print("num of all aux species: ", len(df_aux.species.unique()))
        print("num of train aux species: ", len(df_aux_train.species.unique()))
        print("num of train aux images: ", len(df_aux_train))

    def _get_non_local_ood(self):
        species_to_exclude, speciesKey_to_exclude = self._get_regional_id_and_ood_species()
        # get all australian species
        df_global = pd.read_csv(self.csv_dir + "global_moths_with_names.csv")
        australia_range = {
            "nelat": -10,
            "nelng": 154,
            "swlat": -44,
            "swlng": 113,
        }  # Continental Australia
        lat_min, lat_max = (
            australia_range["swlat"],
            australia_range["nelat"],
        )
        lng_min, lng_max = (
            australia_range["swlng"],
            australia_range["nelng"],
        )

        in_range = (
            (df_global["decimalLatitude"] > lat_min)
            & (df_global["decimalLatitude"] < lat_max)
            & (df_global["decimalLongitude"] > lng_min)
            & (df_global["decimalLongitude"] < lng_max)
        )

        species_australia = set(df_global[in_range].species)
        speciesKey_australia = set(df_global[in_range].speciesKey)
        species_australia = species_australia - species_to_exclude
        speciesKey_australia = speciesKey_australia - speciesKey_to_exclude

        df = df_global[df_global["species"].isin(species_australia)]

        df_test = df.groupby("species", group_keys=False).apply(
            lambda group: group.sample(n=min(len(group), 30), random_state=42)
        )
        df_val = df.loc[~df.index.isin(df_test.index)]  # drop the samples in the test split
        df_val = df_val.sample(n=int(len(df_test) / 10), random_state=42)  # sample 10%

        print("australia val: ", len(df_val))
        print("australia test: ", len(df_test))

        df_test.to_csv(self.csv_dir + "05_test_ood_non-local.csv", index=False)
        df_val.to_csv(self.csv_dir + "05_val_ood_non-local.csv", index=False)

        return df_test, df_val

    def _get_non_moth_ood(self):
        df_val = pd.read_csv(self.ami_binary_dir + "ami-gbif_binary_val.csv")
        df_val = df_val[df_val["binary"] == "nonmoth"]
        df_test = pd.read_csv(self.ami_binary_dir + "ami-gbif_binary_test.csv")
        df_test = df_test[df_test["binary"] == "nonmoth"]
        df_val = df_val.sample(n=3500, random_state=42)
        df_test = df_test.sample(n=35000, random_state=42)

        df_test.to_csv(self.csv_dir + "06_test_ood_non-moth.csv", index=False)
        df_val.to_csv(self.csv_dir + "06_val_ood_non-moth.csv", index=False)

        return df_test, df_val

    def _get_local_ood_regional(self, region):
        df_global = pd.read_csv(self.csv_dir + "global_moths_with_names.csv")
        df_in = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv")

        lat_min, lat_max = (
            self.region_dict[region]["swlat"] - 1,
            self.region_dict[region]["nelat"] + 1,
        )
        lng_min, lng_max = (
            self.region_dict[region]["swlng"] - 3,
            self.region_dict[region]["nelng"] + 3,
        )

        in_range = (
            (df_global["decimalLatitude"] > lat_min)
            & (df_global["decimalLatitude"] < lat_max)
            & (df_global["decimalLongitude"] > lng_min)
            & (df_global["decimalLongitude"] < lng_max)
        )

        # df_global[in_range].to_csv(
        #     self.ood_dir
        #     + f"{self.region_dict[region]['index']:02d}_{region}_in_range.csv",
        #     index=False,
        # )
        species_all = set(df_global[in_range].species)
        species_in = set(df_in.species.unique())
        species_out = species_all - species_in
        df_out = df_global[df_global["species"].isin(species_out)]

        print("region: ", region)
        print("num of id species: ", len(species_in))
        print("num of ood species: ", len(species_out))
        print("num of total ood images: ", len(df_out))

        # sample 90% if there are more than 30 images in that species
        def sampler(group):
            if len(group) > 30:
                return group.sample(frac=0.9, random_state=42)
            return group

        df_test = df_out.groupby("species", group_keys=False).apply(sampler)
        df_val = df_out.loc[~df_out.index.isin(df_test.index)]
        assert len(df_out.species.unique()) == len(df_test.species.unique())

        fn_test = f"{self.region_dict[region]['index']:02d}_{region}_test_ood_local.csv"
        df_test.to_csv(self.csv_dir + fn_test, index=False)

        fn_val = f"{self.region_dict[region]['index']:02d}_{region}_val_ood_local.csv"
        df_val.to_csv(self.csv_dir + fn_val, index=False)

        return df_test, df_val

    def _merge_csv(self, df_img, fn):
        df_taxonomy_map = pd.read_csv(self.csv_dir + "taxonomy_map.csv")
        assert len(set(df_img.speciesKey) - set(df_taxonomy_map.speciesKey)) == 0
        # df_species = df_img.speciesKey.drop_duplicates()
        # num_speciesKey = len(df_species)
        # df_species = pd.merge(df_species, df_species_map, how="left", on="speciesKey")
        # num_species = len(df_species)
        # assert num_speciesKey == num_species  # make sure that this is a bijection
        df_img_merged = pd.merge(df_img, df_taxonomy_map, how="left", on="speciesKey")
        assert len(df_img) == len(df_img_merged)
        df_img_merged.to_csv(self.csv_dir + fn, index=False)

    def _get_regional_id_and_ood_species(self):

        species = set()
        speciesKey = set()

        for region in self.region_dict.keys():
            fn_in = f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv"
            fn_out = f"{self.region_dict[region]['index']:02d}_{region}_test_ood_local.csv"
            df_in = pd.read_csv(self.csv_dir + fn_in)
            df_out = pd.read_csv(self.csv_dir + fn_out)
            species_in = set(df_in.species)
            species_out = set(df_out.species)
            species |= species_in
            species |= species_out

            speciesKey_in = set(df_in.speciesKey)
            speciesKey_out = set(df_out.speciesKey)
            speciesKey |= speciesKey_in
            speciesKey |= speciesKey_out

        return species, speciesKey

    def _compare_taxonomy(self, df_in, df_out):
        genus_in = set(df_in.genus)
        genus_out = set(df_out.genus)

        genus_percent = len(genus_in & genus_out) / len(genus_out)

        family_in = set(df_in.family)
        family_out = set(df_out.family)

        family_percent = len(family_in & family_out) / len(family_out)

        print(f"Overlapping genera (%): {genus_percent * 100:.2f}")
        print(f"Overlapping families (%): {family_percent * 100:.2f}")

    def create_mos_metadata(self):
        filenames = [
            "train_id",
            "val_id",
            "test_id",
        ]
        for region in self.region_dict.keys():
            dfs = [
                pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_{fn}.csv")
                for fn in filenames
            ]
            for fn, df in zip(filenames, dfs):
                self._create_mos_labels(
                    df, f"{self.region_dict[region]['index']:02d}_{region}_{fn}", taxon_name="family"
                )
                self._create_mos_labels(
                    df, f"{self.region_dict[region]['index']:02d}_{region}_{fn}", taxon_name="genus"
                )

    def hop_analysis(self):
        # for region in self.region_dict.keys():
        #     df_id = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv")
        #     df_ood = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_test_ood_local.csv")

        #     id_genera = set(df_id.genus)
        #     id_families = set(df_id.family)

        #     hop1 = df_ood[df_ood["genus"].isin(id_genera)]
        #     hop2 = df_ood[~(df_ood["genus"].isin(id_genera)) & (df_ood["family"].isin(id_families))]
        #     hop3 = df_ood[~(df_ood["family"].isin(id_families))]

        #     print("local")
        #     print(f"total - number of species: {len(df_ood.species.unique())}", f"number of images: {len(df_ood)}")
        #     print(f"hop1 - number of species: {len(hop1.species.unique())}", f"number of images: {len(hop1)}")
        #     print(f"hop2 - number of species: {len(hop2.species.unique())}", f"number of images: {len(hop2)}")
        #     print(f"hop3 - number of species: {len(hop3.species.unique())}", f"number of images: {len(hop3)}")

        df_ood = pd.read_csv(self.csv_dir + "05_test_ood_non-local.csv")

        for region in self.region_dict.keys():
            df_id = pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_train_id.csv")

            id_genera = set(df_id.genus)
            id_families = set(df_id.family)

            hop1 = df_ood[df_ood["genus"].isin(id_genera)]
            hop2 = df_ood[~(df_ood["genus"].isin(id_genera)) & (df_ood["family"].isin(id_families))]
            hop3 = df_ood[~(df_ood["family"].isin(id_families))]

            print("non local")
            print(f"total - number of species: {len(df_ood.species.unique())}", f"number of images: {len(df_ood)}")
            print(f"hop1 - number of species: {len(hop1.species.unique())}", f"number of images: {len(hop1)}")
            print(f"hop2 - number of species: {len(hop2.species.unique())}", f"number of images: {len(hop2)}")
            print(f"hop3 - number of species: {len(hop3.species.unique())}", f"number of images: {len(hop3)}")

    def get_summary(self):
        # # make sure id & local ood & non-local ood & aux are pairwise disjoint
        num_of_species, num_of_images = [], []
        region_list = []
        filenames = [
            # "train_id",
            "train_aux",
            # "val_id",
            # "val_ood",
            # "test_id",
            # "test_ood_local",
        ]

        # df_non_local_val = pd.read_csv(self.csv_dir + "05_val_ood_non-local.csv")
        # df_non_local_test = pd.read_csv(self.csv_dir + "05_test_ood_non-local.csv")
        # df_non_moth_val = pd.read_csv(self.csv_dir + "06_val_ood_non-moth.csv")
        # df_non_moth_test = pd.read_csv(self.csv_dir + "06_test_ood_non-moth.csv")

        # self._convert_to_txt(df_non_local_val, "05_val_ood_non-local.txt", is_ood=True)
        # self._convert_to_txt(df_non_local_test, "05_test_ood_non-local.txt", is_ood=True)
        # self._convert_to_txt(df_non_moth_val, "06_val_ood_non-moth.txt", is_ood=True)
        # self._convert_to_txt(df_non_moth_test, "06_test_ood_non-moth.txt", is_ood=True)

        for region in self.region_dict.keys():
            print(region)
            dfs = [
                pd.read_csv(self.csv_dir + f"{self.region_dict[region]['index']:02d}_{region}_{fn}.csv")
                for fn in filenames
            ]

            with open(
                self.csv_dir
                + f"{self.region_dict[region]['index']:02d}_ami-gbif_fine-grained_{region}_category_map.json",
                "r",
            ) as file:
                category_map = json.load(file)

            for fn, df in zip(filenames, dfs):
                is_ood = "id" not in fn
                self._convert_to_txt(
                    df,
                    f"{self.region_dict[region]['index']:02d}_{region}_{fn}.txt",
                    category_map,
                    is_ood,
                )

                # if not is_ood:
                #     self._create_mos_labels(df, f"{self.region_dict[region]['index']:02d}_{region}_{fn}", category_map)

            # df_dict = {}
            # for i, fn in enumerate(filenames):
            #     df_dict[fn] = dfs[i]

        #     self._compare_taxonomy(df_dict["train_id"], df_dict["test_ood_local"])
        #     self._compare_taxonomy(df_dict["train_id"], df_non_local_test)

        #     column = "species"
        #     assert pairwise_disjoint(
        #         [
        #             set(df_dict["train_id"][column]),
        #             set(df_dict["train_aux"][column]),
        #             set(df_dict["test_ood_local"][column]),
        #             set(df_non_local_test[column]),
        #         ]
        #     )

        #     column = "speciesKey"
        #     assert pairwise_disjoint(
        #         [
        #             set(df_dict["train_id"][column]),
        #             set(df_dict["train_aux"][column]),
        #             set(df_dict["test_ood_local"][column]),
        #             set(df_non_local_test[column]),
        #         ]
        #     )

        #     for fn in df_dict.keys():
        #         num_of_species.append(len(df_dict[fn].speciesKey.unique()))
        #         num_of_images.append(len(df_dict[fn]))

        #     region_list += [region for _ in range(len(filenames))]

        # data_type_list = filenames + filenames + filenames
        # region_list += [
        #     "non-local (australia)",
        #     "non-local (australia)",
        #     "non-moth",
        #     "non-moth",
        # ]

        # data_type_list += ["val_ood", "test_ood", "val_ood", "test_ood"]

        # num_of_images += [
        #     len(df_non_local_val),
        #     len(df_non_local_test),
        #     len(df_non_moth_val),
        #     len(df_non_moth_test),
        # ]

        # num_of_species += [
        #     len(df_non_local_val.speciesKey.unique()),
        #     len(df_non_local_test.speciesKey.unique()),
        #     len(df_non_moth_val.speciesKey.unique()),
        #     len(df_non_moth_test.speciesKey.unique()),
        # ]

        # data = list(zip(region_list, data_type_list, num_of_species, num_of_images))

        # summary = pd.DataFrame(
        #     data,
        #     columns=["region", "dataset_type", "number_of_species", "number_of_images"],
        # )

        # summary.to_csv(self.csv_dir + "summary.csv", index=False)

    def _convert_to_txt(self, df, fn, category_map=None, is_ood=False):
        df = df.fillna(0)
        df = df.astype({"speciesKey": "int64"})
        if is_ood:
            labels = [-1 for _ in range(len(df))]
        else:
            labels = [category_map[str(s)] for s in df.speciesKey]

        lines = [f"{image_path} {label}" for (image_path, label) in zip(df.image_path, labels)]

        with open(self.txt_dir + fn, "w") as f:
            for line in lines:
                f.write(f"{line}\n")

    def two_level_label(self, group_label, class_label):
        label = '{ "group_label": ' + str(group_label) + ' ,"class_label": ' + str(class_label) + " }"
        return label

    def _create_mos_labels(self, df, fn, taxon_name):
        df = df.astype({"speciesKey": "int64"})
        sorted_taxon = sorted(list(df[taxon_name].unique()))
        taxon_map = {categ: id for id, categ in enumerate(sorted_taxon)}
        taxon_label = [taxon_map[s] for s in df[taxon_name]]
        taxon_species_dict = df.groupby(taxon_name)["speciesKey"].apply(set).to_dict()

        species_map = dict()

        for taxon in taxon_species_dict:
            species_map[taxon] = {categ: id for id, categ in enumerate(sorted(list(taxon_species_dict[taxon])))}

        species_label = []

        for species, taxon in zip(list(df["speciesKey"]), list(df[taxon_name])):
            species_label.append(species_map[taxon][species])

        lines = [
            f"{image_path} {self.two_level_label(group_label, class_label)}"
            for (image_path, group_label, class_label) in zip(df.image_path, taxon_label, species_label)
        ]

        with open(self.txt_dir + f"{fn}_mos_{taxon_name}.txt", "w") as f:
            for line in lines:
                f.write(f"{line}\n")


def get_taxonomy_name_from_gbif(species_key, verbose=False):
    url = f"https://api.gbif.org/v1/species/{species_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        species = data.get("species")
        genus = data.get("genus")
        family = data.get("family")
        if verbose:
            print(species, genus, family)
        return (species, genus, family)
    else:
        print("Error:", response.status_code)
        return (None, None, None)
