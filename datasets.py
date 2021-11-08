from torchvision.datasets.vision import *
import torch
from PIL import Image


class Dataset1(VisionDataset):
    def __init__(
        self, 
        root: str, 
        image_set: str = "train",
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        
        if image_set not in ("train", "val"):
            assert "err"
        
        self.objs = list(filter(os.path.isdir, [os.path.join(root, d) for d in os.listdir(root)]))
        self.objs.sort()
        self.images = []
        self.targets = []
        self.target_type = ("idk", "idk2", "idk3")


        for obj in self.objs:
            files = [os.path.join(obj, f) for f in os.listdir(obj)]
            files.sort()
            self.images.append(files[-1])
            self.targets.append(files[:-1])
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """

        image = Image.open(self.images[index]).convert('RGB')
        
        targets = []
        for i, t in enumerate(self.target_type):
            target = Image.open(self.targets[index][i]).convert('P')
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


class Dataset2(VisionDataset):
    def __init__(
        self, 
        root: str, 
        image_set: str = "train",
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        
        dataset_path = {
            "train" : "a. Training Set",
            "val" : "b. Testing Set"
        }

        if image_set not in dataset_path:
            assert "err"


        self.images = os.listdir(os.path.join(root, "A. Segmentation", "1. Original Images", dataset_path[image_set] ) )
        self.images = [ os.path.join(root, "A. Segmentation", "1. Original Images", dataset_path[image_set], f) for f in self.images]
        self.images.sort()

        self.targets = {
            "1. Microaneurysms" : None,
            "2. Haemorrhages"   : None,
            "3. Hard Exudates"  : None,
            "4. Soft Exudates"  : None,
            "5. Optic Disc"     : None,
        }

        for t in self.targets:
            self.targets[t] = os.listdir( os.path.join(root , "A. Segmentation" , "2. All Segmentation Groundtruths" , dataset_path[image_set] , t) )
            self.targets[t] = [ os.path.join(root , "A. Segmentation" , "2. All Segmentation Groundtruths" , dataset_path[image_set] , t, f) for f in self.targets[t]]
            self.targets[t].sort()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for t in self.targets:
            target = Image.open(self.targets[t][index]).convert('P')
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

if __name__ == '__main__':
    from presets import SegmentationPresetTrain
    from matplotlib.pyplot import imshow
    import numpy as np
    trans = SegmentationPresetTrain(520, 480)

    xx = ( Image.new('P', (1000, 1000)) ,Image.new('P', (1000, 1000)) )

    img, target = trans(Image.new('RGB', (1000, 1000)), xx )
    ds1 = Dataset1("../BIO_data/DB_UoA", transform=trans)
    print(len(ds1))
    imshow(np.asarray(ds1[0][1]))
    ds1[0]