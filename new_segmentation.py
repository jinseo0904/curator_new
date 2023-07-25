# dependencies
import monai
import matplotlib
import sys
import SimpleITK as sitk
import os
import itkUtilities as itku
import numpy as np
import pandas as pd
import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    EnsureTyped,
    EnsureType,
    Invertd,
    AddChanneld,
    AsChannelFirstd,
    ToTensord
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import glob
import csv
import nibabel as nb

base_dir = r'D:\nii'
train_images = sorted(os.listdir(r'D:\nii\images'))
images_names = train_images
train_labels = sorted(os.listdir(r'D:\nii\images'))
train_images = [base_dir + '/images/' + file for file in train_images]
train_labels = [base_dir + '/images/' + file for file in train_labels]

print(train_images[1])
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
#print(train_images)
#print(data_dicts)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=['image', 'label']),
        ToTensord(keys=["image", "label"]),
    ]
)

device = torch.device("cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# load best metric (using cpu device)
root_dir = r'D:\TOF'
model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model_2022_11_08_only_all3.pth"), map_location=torch.device('cpu')))

# write csv file
all_fields = ["file-name","area-1-model", "area-2-model", "area-3-model"]
progressFilePath = os.path.join(root_dir, "cine_areas.csv")
if not os.path.isfile(progressFilePath):
    with open(progressFilePath, "w") as outputFile: 
        writer = csv.DictWriter(outputFile, lineterminator='\n', fieldnames=all_fields)
        writer.writeheader()

#Resized Images
test_ds = Dataset(data=data_dicts, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=100)
print(type(test_loader))
model.eval()


sample = nb.load(r'D:\data-cine\images\img0001-114.487.dcm.nii.gz')

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        print(type(test_data["image"]))
        for k in range(0, 2):
            roi_size = (256, 256)
            sw_batch_size = 1
            
            test_outputs = sliding_window_inference(
                test_data["image"].to(device), roi_size, sw_batch_size, model
            )
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"image {k+1}")
            plt.imshow(test_data["image"][k][0], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title(f"output {k+1}")
            plt.imshow(torch.argmax(
            test_outputs, dim=1).detach().cpu()[k])
            plt.show()

            slices = []
            tosave = torch.argmax(test_outputs, dim=1).detach().cpu()[k]
            slices.append(tosave.numpy())
            slices = np.asarray(slices).swapaxes(0,2)
            print(slices.shape)
            print(sample.affine)

            copynifti = nb.Nifti1Image(slices, sample.affine, sample.header)
            print(copynifti.shape)
            output_dir = r'D:\TOF'
            name = '{}_segmented.nii'.format(k)
            nb.save(copynifti, os.path.join(output_dir, name))
            print('saved to {}'.format(os.path.join(output_dir, name)))


