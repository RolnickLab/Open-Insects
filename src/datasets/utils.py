def convert_to_txt(df, fn, closed_set_map=None):
    df = df.fillna(0)
    df = df.astype({"speciesKey": "int64"})
    df.speciesKey

    # create aux_species_map
    aux_species = set() - set(closed_set_map.keys())
    aux_species = sorted(list(aux_species))
    aux_species_map = {str(categ): id + len(closed_set_map) for id, categ in enumerate(aux_species)}

    print(aux_species_map)

    # labels = None
    # lines = [f"{image_path} {label}" for (image_path, label) in zip(df.image_path, labels)]

    # with open(fn, "w") as f:
    #     for line in lines:
    #         f.write(f"{line}\n")
