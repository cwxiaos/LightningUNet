import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dataset")

args = parser.parse_args()

if __name__ == "__main__":
    # assert torch.cuda.is_available(), f"CUDA is essential"
    device = torch.device("cuda")
