from torchvision.datasets.vision import *
import torch
from PIL import Image


class Dataset0(VisionDataset):
    def __init__(
        self, 
        root: str, 
        image_set: str = "train",
        transforms: Optional[Callable] = None, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        
        image_sets = {
            "train" : "Training-data",
            "val" : "Testing-data"
        }

        if image_set not in image_sets:
            assert "err"
        
        dataset_dir = os.path.join(root, image_sets[image_set])
        default_txt = os.path.join(dataset_dir, 'ImageSets','Segmentation','default.txt')

        self.objs = []
        with open(default_txt, 'r') as f:
            self.objs = [s.rstrip('\n') for s in f.readlines()]

        self.images = []
        self.targets = []

        for obj in self.objs:
            tmp = 'JPEGImages'
            tmp2 = 'SegmentationClass'
            self.images.append(os.path.join(dataset_dir, tmp, obj+'.jpg'))
            self.targets.append(os.path.join(dataset_dir, tmp2, obj+'.png'))
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index]).convert('RGB')

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target[2] = (1.0 * (1.0-target[0]) * (1.0-target[1]))

        target = target.argmax(0)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

if __name__ == '__main__':
    from presets import SegmentationPresetTrain
    from matplotlib.pyplot import imshow
    import numpy as np
    trans = SegmentationPresetTrain(520, 480)

    i = Image.new('RGB', (1000, 1000))
    xx = ( Image.new('P', (1000, 1000)) ,Image.new('P', (1000, 1000)) )

    img, target = trans( i, xx )
    imshow(img[0])
    ds1 = Dataset0(os.path.join('..', 'BIO_data', 'RetinaDataset'), transforms=trans)
    print(len(ds1))
    item = ds1[0]
    imshow(np.asarray(item[0][0]))
    ds1[0]