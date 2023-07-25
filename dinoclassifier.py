#from asyncio.windows_events import NULL
from PIL import Image as im
import os
import csv
import logging
import glob
import re
from tqdm import tqdm
import shutil
import collections
from datetime import datetime
logger=logging.getLogger("curator")

# load dependencies
import pandas as pd
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import matplotlib.image
import pydicom

# packages for neural network
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader

# define names and corresponding dictionaries
orientation_names = ['2CH','3CH','4CH','AOFLOW','APEX','AXIAL','AXIAL_MIP','BASE','CORONAL_MIP','DAO','IVC','LOC',
                     'LPA','LPV','LVOT','MID','MPA','RPA','RPV','RV2CH','RVOT','SAGITTAL_MIP','SAG_RVOT','SAX','SVC','TRICUSPID']

contrast_dict = {'CINE': 0, 'DE': 1, 'HASTE': 2, 'PC_MAG': 3, 'PC_PHASE': 4, 'PERF_AIF': 5, 'PSIR_MAG': 6, 'PSIR_PHASE': 7,
                  'SCOUT': 8, 'T1': 9, 'T1*': 10, 'T1RHO': 11, 'TRUFI': 12, 'TWIST': 13}
contrast_names = list(contrast_dict.keys())

orientation_dict = dict(zip(orientation_names, range(len(orientation_names))))

# define maps for later use
orientation_map = {v: k for k, v in orientation_dict.items()}
contrast_map = {v: k for k, v in contrast_dict.items()}

# to check for single / multiple subjects
def single_subject(input_dir):
    folders = os.listdir(input_dir)
    checkpath = os.path.join(input_dir, folders[0])
    single = False

    for i in os.listdir(checkpath):
        if i.endswith('.dcm'):
            single = True
    return single

# callee from curator.py
def classifyDicom(input_dir, output_dir, use_dicom=False, sax_cine_only=False, disable_sorted_folders = False):
    logger.info("classifier.classifyDicom")

    # check if output path already exists
    date = 'dcm_' + str(datetime.now().month)
    date += "_" + str(datetime.now().day)
    date += "_" + str(datetime.now().hour)
    date += "_" + str(datetime.now().minute)


    outpath = os.path.join(output_dir, date)
    if not(os.path.isdir(outpath)):
        os.mkdir(outpath)

    temp = os.path.join(outpath, 'temp')
    if not(os.path.isdir(temp)):
        os.mkdir(temp)

    # initialize torch models here
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('       using device type: {}'.format(device))
    oripath = os.path.join(os.getcwd(), 'models/orientation.pt')
    conpath = os.path.join(os.getcwd(), 'models/contrast.pt')

    ori_model = OrientationClassifier()
    ori_model.load_state_dict(torch.load(oripath, map_location=device))
    ori_model.eval()

    con_model = ContrastClassifier()
    con_model.load_state_dict(torch.load(conpath, map_location=device))
    con_model.eval()

    ori_model.to(device)
    con_model.to(device)


    onesubject = single_subject(input_dir)
    subject_labels, seriesnames, ori_preds, con_preds, ori_probs, con_probs, sax, cine = ([] for _ in range(8))

    if onesubject:
        logger.info('       working with only one subject')
        subject_path = os.path.join(temp, os.path.basename(input_dir))
        if not(os.path.isdir(subject_path)):
            os.mkdir(subject_path)


        sir, or_p, con_p, ori_pr, con_pr, sax_pr, cine_pr = classifySubject(input_dir, outpath, subject_path, ori_model, con_model, device, use_dicom, sax_cine_only, disable_sorted_folders)
        subject_labels += [os.path.basename(input_dir) for _ in range(len(sir))]
        seriesnames += sir
        ori_preds += or_p
        con_preds += con_p
        ori_probs += ori_pr
        con_probs += con_pr
        sax += sax_pr
        cine += cine_pr

    else:
        logger.info('       working with multiple subjects')

        # run classifier for each subject
        subjects = os.listdir(input_dir)
        for subject in subjects:

            # make subjects temp folder
            subject_path = os.path.join(temp, subject)
            if not(os.path.isdir(subject_path)):
                os.mkdir(subject_path)

            singleinput = os.path.join(input_dir, subject)
            sir, or_p, con_p, ori_pr, con_pr, sax_pr, cine_pr = classifySubject(singleinput, outpath, subject_path, ori_model, con_model, device, use_dicom, sax_cine_only, disable_sorted_folders)
            subject_labels += [subject for _ in range(len(sir))]
            seriesnames += sir
            ori_preds += or_p
            con_preds += con_p
            ori_probs += ori_pr
            con_probs += con_pr
            sax += sax_pr
            cine += cine_pr
    
    # write predictions to csv file
    with open(os.path.join(outpath,'predictions.csv'),"w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject Name', 'File Name', 'Orientation', 'Contrast', 'Orientation %', 'Contrast %', "Sax %", "Cine %"])

        for i, name in enumerate(seriesnames):
            writer.writerow([subject_labels[i], name, ori_preds[i], con_preds[i], ori_probs[i], con_probs[i], sax[i], cine[i]])

def preprocess_image(img):
    img *= (255.0/(np.amax(img)))
    reverse = 255.0 - img
    return reverse

def process_series(input_dir, temp_dir, ori_model, con_model, device):
    seriesnames, orientation_predictions, contrast_predictions, ori_certainty, con_certainty, sax_prob, cine_prob = ([] for i in range(7))

    # iterate over each series of dicoms
    for series in os.listdir(input_dir):
        
        # make a corresponding jpg folder, where converted jpgs will be saved
        series_dir = os.path.join(temp_dir, series)
        if not(os.path.isdir(series_dir)):
            os.mkdir(series_dir)

        dicomfolder = os.path.join(input_dir, series)

        # only process if it's a folder containing dicoms
        try:
            if os.path.isdir(dicomfolder):
                logger.info('               currently processing series:   {}'.format(series))

                for dicom in os.listdir(dicomfolder):
                    if dicom.endswith('.dcm'):
                        img = pydicom.read_file(os.path.join(dicomfolder, dicom)).pixel_array.astype('float64')
                        rgb = preprocess_image(img)

                        # should make another layer of directory in order to initialize Imagefolder
                        inside_folder = os.path.join(series_dir, 'jpegs')
                        if not(os.path.isdir(inside_folder)):
                            os.mkdir(inside_folder)

                        dest = inside_folder + '/' + dicom[:-4] + '.jpg'
                        matplotlib.image.imsave(dest, rgb, cmap='binary')

                # now, use the folder to create predictions:
                test_data = datasets.ImageFolder(root=series_dir, transform=eval_transform)
                test_dataloader = DataLoader(dataset=test_data,
                                        batch_size=1,
                                        num_workers=2,
                                        shuffle=False) # don't usually need to shuffle testing data
                
                with torch.no_grad():
                    ori_predictions = []
                    con_predictions = []

                    for images, _ in test_dataloader:
                        model_input = images.to(device)

                        outputs = ori_model(model_input)

                        classifications = torch.argmax(outputs, dim=1)
                        ori_predictions += list(map(orientation_map.get, classifications.tolist()))
                        #print(list(map(orientation_map.get, classifications.tolist())))

                        # contrast
                        outputs = con_model(model_input)
                        classifications = torch.argmax(outputs, dim=1)
                        #print(list(map(contrast_map.get, classifications.tolist())))
                        con_predictions += list(map(contrast_map.get, classifications.tolist()))

                ori_output, or_count = collections.Counter(ori_predictions).most_common(1)[0]
                con_output, con_count = collections.Counter(con_predictions).most_common(1)[0]
                seriesnames.append(series)
                orientation_predictions.append(ori_output)
                contrast_predictions.append(con_output)
                ori_certainty.append(or_count / len(ori_predictions))
                con_certainty.append(con_count / len(con_predictions))

                # get SAX and CINE probability for ROC estimation
                sax_counts = ori_predictions.count("SAX")
                cine_counts = con_predictions.count("CINE")
                sax_prob.append(sax_counts / len(ori_predictions))
                cine_prob.append(cine_counts / len(con_predictions))
        except:
            logger.info('                   WARNING: Could not process the series and skipping:   {}'.format(series))

    return seriesnames, orientation_predictions, contrast_predictions, ori_certainty, con_certainty, sax_prob, cine_prob
    

# main classifier function
def classifySubject(input_dir, output_dir, temp_dir, ori_model, con_model, device, use_dicom, sax_cine_only, disable_sorted_folders):
    logger.info('           currently processing patient:   {}'.format(os.path.basename(input_dir)))
    
    if (use_dicom == True):
        # process series in a single subject folder
        seriesnames, orientation_predictions, contrast_predictions, ori_certainty, con_certainty, sax_pr, cine_pr = process_series(input_dir, temp_dir, ori_model, con_model, device)

        # sort folder based on subject
        if not(disable_sorted_folders):
            generate_sorted_folder(input_dir, output_dir, orientation_predictions, contrast_predictions, seriesnames, sax_cine_only)
        return seriesnames, orientation_predictions, contrast_predictions, ori_certainty, con_certainty, sax_pr, cine_pr

    else:
        # load_niftis:
        img_array, phase_counts, fetched_directory, errors = load_niftis(input_dir)
        return [], [], []

def generate_sorted_folder(input_dir, output_dir, orientation, contrast, seriesnames, sax_cine_only):
    # generate a sorted folder in the parent directory
    newfolder = 'Classified_{}'.format(os.path.basename(input_dir))
    newpath = os.path.join(output_dir, newfolder)

    if not(os.path.isdir(newpath)):
        os.mkdir(newpath)

    # iterate through each series
    for i, name in enumerate(seriesnames):
        if not(sax_cine_only) or (sax_cine_only and orientation[i] == 3 and contrast[i] == 0):
            classname = '{}_{}'.format(orientation[i], contrast[i])
            
            # if this class name folder does not exist, make a new one
            classfolder = os.path.join(newpath, classname)
            if not(os.path.isdir(classfolder)):
                os.mkdir(classfolder)

            # copy paste all series
            src = os.path.join(input_dir, name)
            dst = os.path.join(classfolder, name)
            shutil.copytree(src, dst)


class OrientationClassifier(nn.Module):
    def __init__(self):
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        super(OrientationClassifier, self).__init__()
        self.transformer = dinov2_vitb14

        # Make the layers not trainable
        self.transformer.requires_grad = False

        self.classifier = nn.Sequential(
            # change the linear input size accordingly
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(orientation_names))
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
    
class ContrastClassifier(nn.Module):
    def __init__(self):
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        super(ContrastClassifier, self).__init__()
        self.transformer = dinov2_vitb14

        # Make the layers not trainable
        self.transformer.requires_grad = False

        self.classifier = nn.Sequential(
            # change the linear input size accordingly
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, len(contrast_names))
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
    
eval_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(252, 252)),
        # Flip the images randomly on the horizontal
        #transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])
    
# nifti functionalities (needs to be fixed)
def int_code(x):
    return int(re.match('[0-9]+', os.path.basename(x)).group(0))

def load_niftis(input_dir):
    search = sorted(glob.glob(os.path.join(input_dir,'**/*.nii'),recursive=True))
    results = sorted(search, key = int_code)

    # read image files recursively from image directory
    directory = []
    for file in results:
        directory.append((file, os.path.dirname(file)))

    # check for empty folder edge case
    if len(directory) == 0:
        logger.error("classifier was unable to fetch any images from input directory")
        return

    img_array = []
    phase_counts = []
    fetched_directory = []
    errors = 0
    for imgfile, folderpath in tqdm(directory):
        try:
            img = nb.load(imgfile).get_fdata()
        except Exception as e:
            # print(e)
            logging.warning('      failed to convert nifti to numpy array : %s',imgfile)
            errors += 1
        else:
            if len(img.shape) == 4:
                phase_counts.append(img.shape[3])
            else:
                phase_counts.append(-1)
            img_array.append(preprocess_image(img, imgfile))
            fetched_directory.append((imgfile, folderpath))
    img_array = np.array(img_array)
    fetched_directory.sort(key=lambda y: y[1])
    return img_array, phase_counts, fetched_directory, errors