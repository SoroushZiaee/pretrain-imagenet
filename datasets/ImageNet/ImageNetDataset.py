import os
from torchvision.datasets.utils import extract_archive, verify_str_arg


from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder, default_loader

META_FILE = "meta.bin"


class ImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        split: str = "train",
        dst_meta_path: str = os.path.join(os.getcwd(), "data", "ImageNet"),
        **kwargs: Any,
    ):
        root = self.root = os.path.expanduser(root)
        self.dst_meta_path = dst_meta_path
        os.makedirs(dst_meta_path, exist_ok=True)
        self.split = verify_str_arg(split, "split", ("train", "validation"))

        self.parse_archives()
        wnid_to_classes = self.load_meta_file()[0]

        self.split_folder = os.path.join(root, split)
        super().__init__(
            self.split_folder,
            **kwargs,
        )
        self.root = root
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self) -> Tuple[Dict[str, str], List[str]]:
        file = os.path.join(self.dst_meta_path, META_FILE)
        return torch.load(file, weights_only=True)

    def parse_archives(self) -> None:
        if not os.path.exists(os.path.join(self.dst_meta_path, META_FILE)):
            self.__parse_devkit_archive(dst_meta_path=self.dst_meta_path)

    def __parse_devkit_archive(self, dst_meta_path: str):

        import scipy.io as sio

        def parse_meta_mat(
            devkit_root: str,
        ) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
            metafile = os.path.join(devkit_root, "data", "meta.mat")
            meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
            nums_children = list(zip(*meta))[4]
            meta = [
                meta[idx]
                for idx, num_children in enumerate(nums_children)
                if num_children == 0
            ]
            idcs, wnids, classes = list(zip(*meta))[:3]
            classes = [tuple(clss.split(", ")) for clss in classes]
            idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
            wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
            return idx_to_wnid, wnid_to_classes

        def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
            file = os.path.join(
                devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt"
            )
            with open(file) as txtfh:
                val_idcs = txtfh.readlines()
            return [int(val_idx) for val_idx in val_idcs]

        devkit_root = os.path.join(self.root, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
        torch.save((wnid_to_classes, val_wnids), os.path.join(dst_meta_path, META_FILE))
