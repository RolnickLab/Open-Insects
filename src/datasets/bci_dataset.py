import pandas as pd


def has_image(id, file_list):
    for file in file_list:
        if id in file:
            return True
    return False


df_bold = pd.read_csv("src/datasets/BCI_Leps_BATCHID_dec24.csv")
df_gdrive = pd.read_csv("src/datasets/files.csv")

BCI_samples = []
for sample in df_bold.Sampleid:
    if "BCI" in sample:
        BCI_samples.append(sample)

# print(len(BCI_samples))
# print(BCI_samples)

images = list(df_gdrive.image_path)

for sample in BCI_samples:
    if has_image(sample, images):
        print(sample)
