import os
import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed


def process_images(image_path, image, transform, output_split_dir):
    image_name = image.split(".")[0]
    image = PIL.Image.open(os.path.join(image_path, f"{image_name}.jpg")).convert("RGB")
    image = transform(image)

    # Save the preprocessed image
    torch.save(image, os.path.join(output_split_dir, f"{image_name}.pt"))

    return [f"{image_name}.jpg", f"{image_name}.pt"]


def preprocess_and_save_images(root, output_dir, transform):
    image_path = os.path.join(root, "images")
    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(image_path)
    meta = Parallel(n_jobs=15)(
        delayed(process_images)(image_path, image, transform, output_dir)
        for image in tqdm(images, total=len(images))
    )

    meta = pd.DataFrame(meta, columns=["image_path", "target_path"])
    meta["image_idx"] = meta["image_path"].apply(lambda x: int(x.split(".")[0]))
    meta.sort_values("image_idx", inplace=True)
    meta.reset_index(drop=True, inplace=True)
    meta.to_csv(os.path.join(output_dir, "meta.csv"), index=False)

    splits_path = os.path.join(root, "splits")
    split_output_path = os.path.join(output_dir, "splits")
    os.makedirs(split_output_path, exist_ok=True)

    for split in tqdm(os.listdir(splits_path)):
        temp = pd.read_csv(
            os.path.join(splits_path, split),
            delim_whitespace=True,
            header=None,
        )
        temp.columns = ["image_name", "memo_score"]
        temp["preprocessed_path"] = temp["image_name"].apply(
            lambda x: f"{x.split('.')[0]}.pt"
        )
        data_path = os.path.join(split_output_path, f"{split.split('.')[0]}.csv")
        temp.to_csv(data_path, index=False)


def main():
    root = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/lamem_images/lamem"
    output_dir = "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/LaMem/preprocessed/dataset"
    mean = np.load(
        "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/datasets/LaMem/support_files/image_mean.npy"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), PIL.Image.BILINEAR),
            lambda x: np.array(x),
            lambda x: np.subtract(x[:, :, [2, 1, 0]], mean),  # Subtract average mean
            lambda x: x[15:242, 15:242],  # Center crop
            transforms.ToTensor(),
        ]
    )

    preprocess_and_save_images(
        root,
        output_dir,
        transform,
    )


if __name__ == "__main__":
    main()
