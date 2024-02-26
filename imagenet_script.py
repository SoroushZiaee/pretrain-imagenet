from datasets.ImageNet.ImageNetDataset import ImageNet

if __name__ == "__main__":
    root = (
        "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/data/ImageNet"
    )
    dataset = ImageNet(root=root, split="train")
