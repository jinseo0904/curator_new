
case = 'TOF-003'

import sys
import SimpleITK as sitk
import os
import itkUtilities as itku
import numpy as np
import pandas as pd
import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
import itk

import sys
import SimpleITK as sitk

import argparse
import nibabel as nb
import csv


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
import os
import glob
import numpy as np
import pydicom


if __name__ == "__main__":

    default_include_tags_txt = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "include_tags_default.txt")

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Beth's Super Buggy Code")
    parser.add_argument("--recreate_nii",action='store_true',help="recreate nii files")

    args = parser.parse_args()

    basedir = 'D:\ismrm\Classified_{}/SAX_CINE'.format(case)
    dataDir = os.path.join('D:\ismrm/', 'data-cine_{}'.format(case))
    segfolder = 'D:\ismrm\Classified_{}'.format(case)
    segDir = os.path.join(segfolder, 'segmentations')


    if not os.path.exists(segDir):
        os.mkdir(segDir)
    cfolders = os.listdir(basedir)

    for cinefolder in cfolders:
        print(cinefolder)
        imgdir = os.path.join(basedir, cinefolder)
        allfiles = os.listdir(imgdir)


    if(args.recreate_nii):
        if not os.path.exists(dataDir):
            os.mkdir(dataDir)
            os.mkdir(os.path.join(dataDir,'images'))
            os.mkdir(os.path.join(dataDir,'labels'))

        cfolders = os.listdir(basedir)
        # print(cfolders)
        for cinefolder in cfolders:
            print('processing folder {}'.format(cinefolder))

            #first get the image stuff for doing the nifti conversion
            imgdir = os.path.join(basedir, cinefolder)
            # print("image directory")
            # print(imgdir)
            targetimgdir = os.path.join(dataDir, 'images')
            targetlabeldir = os.path.join(dataDir, 'labels')
            allfiles = os.listdir(imgdir)
            #print(allfiles)


            for file in allfiles:
                # print("file")
                # print(file)
                imgfile = os.path.join(imgdir, file)
                # print(imgfile)
                itkimg = sitk.ReadImage(imgfile, imageIO="GDCMImageIO")
                itkImgView = sitk.GetArrayFromImage(itkimg)
                raiImg = sitk.GetImageFromArray(itkImgView[0,:,:])
                outsizeforlater = raiImg.GetSize()
                outsize = raiImg.GetSize()
                templist = list(outsize)
                templist[0] = 256
                templist[1] = 256
                outsize = tuple(templist)

                raiImg = itku.resizeImage(raiImg, outsize, interpolation="Linear")
                imgview = raiImg
                segview = sitk.GetImageFromArray(np.ones((256, 256)))

                sitk.imwrite(imgview, os.path.join(targetimgdir, file + '.nii.gz'))
                sitk.imwrite(segview, os.path.join(targetlabeldir, file + '.nii.gz'))

    print('finished processing images')

                



    '''
    train_images = sorted(os.listdir(os.path.join(dataDir,'images')))
    images_names = train_images
    train_labels = sorted(os.listdir(os.path.join(dataDir,'labels')))
    train_images = [os.path.join(dataDir,'images') + "/" + file for file in train_images]
    train_labels = [os.path.join(dataDir,'labels') + "/" + file for file in train_labels]

    #print(train_images)
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    # print(train_images)
    # print(data_dicts)

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=['image', 'label']),
            ToTensord(keys=["image", "label"]),
        ]
    )

    device = torch.device('cpu')


    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)




    root_dir = '/mnt/c/users/bethw/downloads/'
    model.load_state_dict(torch.load(
        os.path.join(root_dir, "best_metric_model_2022_11_08_only_all3.pth"), map_location=torch.device('cpu')))




    # import csv
    # all_fields = ["file-name","area-1-model", "area-2-model", "area-3-model"]
    # root_dir = '/content/drive/MyDrive/TOF/'



    #Resized Images
    test_ds = Dataset(data=data_dicts, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=100)

    model.eval()
    with torch.no_grad():
        with open(os.path.join(dataDir, 'segmentations.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file path', 'trigger time', 'slice location', 'RV blood pool', 'LV myocardium', 'LV blood pool', 'voxel size'])
            for i, test_data in enumerate(test_loader):
                for k in range(0, len(images_names) - 1):
                    roi_size = (256, 256)
                    sw_batch_size = 1

                    type(test_data["image"])
                    
                    test_outputs = sliding_window_inference(
                        test_data["image"].to(device), roi_size, sw_batch_size, model
                    )

                    output = torch.argmax(test_outputs, dim=1).detach().cpu()[k]
                    #print(resized_output)
                    
                    #get dicom info 
                    dicom = pydicom.dcmread(os.path.join(imgdir, allfiles[k]))
                    voxelsize = dicom.PixelSpacing[0] * dicom.PixelSpacing[1] * max(dicom.SliceThickness, dicom.SpacingBetweenSlices)
                    dim1 = dicom.pixel_array.shape[1]
                    dim2 = dicom.pixel_array.shape[0]
                    ddim = (dim1, dim2)
                    resized_output = itku.resizeImage(itk.GetImageFromArray(output.numpy().astype('uint8')), ddim, interpolation="Label")
                    
                    # write entries into csv file

                    row = [images_names[k], dicom.TriggerTime, dicom.SliceLocation]
                    voxelsize = dicom.PixelSpacing[0] * dicom.PixelSpacing[1] * max(dicom.SliceThickness, dicom.SpacingBetweenSlices)
                    
                    label = resized_output.astype(np.int16)
                    counts = [np.count_nonzero(label == 300), np.count_nonzero(label == 200), np.count_nonzero(label == 100)]

                    #filename = images_names[k]


                    row.extend([counts[0] * voxelsize, counts[1] * voxelsize, counts[2] * voxelsize, voxelsize])
                    writer.writerow(row)

                
                
                
                # itk.imwrite(output_img, os.path.join(segDir, '{}.nii.gz'.format(images_names(k)))) #need to change this to have the same filename as from images, testing with k

            

    # #Images with original size
    # count = 0
    # model.eval()
    # with torch.no_grad():
    #     for i, test_data in enumerate(test_loader):
    #       roi_size = (256, 256)
    #       sw_batch_size = 1
    #       test_outputs = sliding_window_inference(test_data["image"].to(device), roi_size, sw_batch_size, model)
    #       for k in range(0, 28):
    #           filename = images_names[count]
    #           count = count + 1

    #           output = torch.argmax(test_outputs, dim=1).detach().cpu()[k]
    #           resized_output = itku.resizeImage(itk.GetImageFromArray(output.numpy().astype('uint8')), (200, 256), interpolation="Label")
    #           image = test_data["image"][k][0]
    #           resized_image = itku.resizeImage(itk.GetImageFromArray(image), (200, 256), interpolation="Linear")

    #           model_areas = [None, None, None]
            
    #           for n in range (0, 3):
    #             print(n)
    #             model_areas[n] = np.count_nonzero(itk.GetArrayFromImage(resized_output) == n+1)

    #           with open(progressFilePath, "a") as outputFile: 
    #                         writer = csv.DictWriter(outputFile, lineterminator='\n', fieldnames=all_fields)
    #                         writer.writerow({"file-name": filename, "area-1-model": model_areas[0], "area-2-model": model_areas[1], "area-3-model": model_areas[2]})
            
    #           plt.figure("check", (18, 6)) 
    #           plt.subplot(1, 2, 1)
    #           plt.title(filename) 
    #           plt.imshow(resized_image, cmap="gray")
    #           plt.subplot(1, 2, 2)
    #           plt.title(f"output {k+1}")
    #           plt.imshow(resized_output)
    #           plt.show()
    '''

