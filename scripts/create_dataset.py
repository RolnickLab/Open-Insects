import argparse
from src.datasets.ami_dataset import AMIDataset
from src.datasets.aux_ablation import AuxDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="aux_ablation")
    parser.add_argument("--metadata_dir")
    parser.add_argument("--ami_dir")
    parser.add_argument("--ami_binary_dir")
    args = parser.parse_args()
    ami_dataset = AMIDataset(args.ami_dir, args.ami_binary_dir, args.metadata_dir)
    ami_dataset.create_aux_dataset()
    aux_dataset = AuxDataset(args.metadata_dir)
    # aux_dataset.get_info()
    # aux_dataset.sample_species_and_images()
