from datasets import Dataset1, Dataset2
from presets import SegmentationPresetTrain, SegmentationPresetEval
from matplotlib.pyplot import imshow
import utils
import torch
import numpy as np

ds2 = Dataset2("../BIO_data/Database", image_set='train', transforms=SegmentationPresetTrain(520, 480))
print(len(ds2))
img, target = ds2[0]

logger = utils.MetricLogger()
train_sampler = torch.utils.data.RandomSampler(ds2)
data_loader = torch.utils.data.DataLoader(
        ds2, batch_size=1,
        sampler=train_sampler, num_workers=4,
        collate_fn=utils.collate_fn, drop_last=True)


for i in data_loader:
    print(i[0].shape)
