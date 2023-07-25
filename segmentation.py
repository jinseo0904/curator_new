#Beth updating Jin's code to perform 2D segmentation, summarizing all data by scan/patient
#3/23/23 update, need to change the way the series data is organized bc of issues with phase indexing
#5/2/23 updating to add additional IDPs strain, peak dV/dt, flow in first third of systole, and contraction fraction
#5/19/23 updating dv/dt calculation to be most negative value so that it is absolute maximum dv/dt during systole

# system
import argparse
import os
import sys
import logging
import glob
import csv

# required modules
import matplotlib.pyplot as plt
import torch 
import nibabel as nb
import numpy as np
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import pydicom
import cv2
import pandas

# logger initialization
logger=logging.getLogger("curator")

#debugging
from PIL import Image

def initLogger(name,logfile):
    # Create debugging information (logger)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(logfile)
    streamHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger

def segment_nii(cine, model): #beth is skipping updating this section now, focusing on dicoms
    # adjust dimensions
    cine_arr = cine.get_fdata()
    roi_size = (256, 256)
    sw_batch_size = 1
    # get slices
    slices = []
    for phase in range(cine_arr.shape[3]):
        slice = torch.from_numpy(cine_arr[:,:,0,phase]).float().unsqueeze(0).unsqueeze(0)
        output = torch.argmax(sliding_window_inference(slice, roi_size, sw_batch_size, model), dim=1).detach()
        slices.append(output.numpy())

    slices = np.transpose(np.asarray(slices)).swapaxes(0,1)

    # make a copy of original nifti with the segmented ventricle
    copynifti = nb.Nifti1Image(slices, cine.affine, cine.header)
    # assert copynifti.header == cine.header
    # print(copynifti.shape)
    # assert copynifti.shape == cine.shape
    return copynifti

def segment_dicom(cine, model): #has been updated by beth 3/10/23
    
    slice = cine.pixel_array.astype('float64') #read in the dicom pixel data for the slice
    og_size_arr = np.shape(slice) #store the original size of the slice
    og_size = (og_size_arr[1], og_size_arr[0]) #swap dimensions because image dimensions are given in width x height (ie columns x rows) rather than rows x columns
    slice = cv2.resize(slice, dsize=(256, 256)) #resize to 256 by 256 before applying the model
    slice = torch.from_numpy(slice.copy()).float().unsqueeze(0).unsqueeze(0) #convert to torch tensor
    roi_size = (256, 256)
    sw_batch_size = 1
    result = sliding_window_inference(slice, roi_size, sw_batch_size, model) #apply the model
    output = torch.argmax(result, dim=1).detach()
    output = np.transpose(output[0,:,:].numpy()) * 100 #bring back to a numpy array
    output = np.rot90(output, 3) #things get rotated in pytorch, so need to rotate and flip
    output = np.fliplr(output) 
    output = cv2.resize(output, dsize = og_size, interpolation = cv2.INTER_NEAREST) #nearest neighbor interpolation to preserve segmentation labels
    return output

# checks input directory hierarchy - is there a more elegant way to do this, by reading the accession / id information in the dicoms?
def single_subject(input_dir):
    folders = os.listdir(input_dir)
    checkpath = os.path.join(input_dir, folders[0])
    single = False

    for i in os.listdir(checkpath):
        if i.endswith('.dcm'):
            single = True
    return single


def processSeries(input_dir, output_dir, use_dicom):

    # find all cine images
    if (use_dicom == True):
        cines = []
        foldernames = []
        skipped = 0

        # check all series that are four dimensional and have more than 20 phases - a baseline check that the inputs are cines
        for folder in os.listdir(input_dir):
            folderpath = os.path.join(input_dir, folder)
            if glob.glob(os.path.join(folderpath,'*.dcm')):
                onlyfiles = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.dcm')]
                #fourdimension = all([pydicom.read_file(f).SeriesInstanceUID == pydicom.read_file(onlyfiles[0]).SeriesInstanceUID for f in onlyfiles])
                
                if len(onlyfiles) > 20: #beth removing fourdimension on 5/4 because I don't know why it was reading false for some of the sax cines in tof-047
                    cines.append(folderpath)
                    foldernames.append(folder)
                else:
                    skipped += 1
        logger.info('               Skipped {0} images, Segmenting {1} cines'.format(skipped, len(cines)))
    else:
        images = glob.glob(os.path.join(input_dir,'*.nii')) + glob.glob(os.path.join(input_dir,'*.nii.gz'))
        cines = []
        skipped = 0
        for nifti_path in images:
            nifti = nb.load(nifti_path)
            if len(nifti.shape) != 4 or nifti.shape[3] < 20:
                skipped += 1
            else:
                cines.append((nifti, nifti_path))
        logger.info('       Skipped {0} images, Segmenting {1} cines'.format(skipped, len(cines)))

    # load monai UNet model
    device = torch.device("cpu")
    model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    )
    model.load_state_dict(torch.load('segmentation_model/20230504-143324_best_metric_model_2d.pth', map_location=torch.device('cpu'))) #the location changes if we are on pmacs

    # generate predictions for each cine
    if (use_dicom == True):
        with open(os.path.join(output_dir, 'segmentations.csv'), 'w', newline='') as f: #open the csv for the scan/subject
            writer = csv.writer(f)
            writer.writerow(['file path', 'trigger time (ms)', 'slice location (mm)', 'R-R (ms)', 'RV blood pool (pixels)', 'LV myocardium (pixels)', 'LV blood pool (pixels)', 'voxel size (mm^3/pixel)']) 
            for (cinefolder, nameonly) in zip(cines, foldernames): #for each series in the patient scan
                # create a directory for each series
                savepath = os.path.join(output_dir, nameonly)
                if not(os.path.exists(savepath)):
                    os.makedirs(savepath)

                # segment each dicom slice
                for dcm in os.listdir(cinefolder):
                    dicom = pydicom.dcmread(os.path.join(cinefolder, dcm))
                    segmented = segment_dicom(dicom, model)
                    assert segmented.shape == dicom.pixel_array.shape
                    dicom.PixelData = segmented.astype(np.int16).tobytes()
                    dicom.SeriesNumber = dicom.SeriesNumber + 3000
                    dicom.save_as(os.path.join(savepath, dcm))
                    if "NominalInterval" in dicom:
                        rr_int = dicom.NominalInterval
                        if rr_int == 0 and "HeartRate" in dicom: #if the nominal interval is zero but there is a heart rate,
                            hr = dicom.HeartRate
                            rr_int = 60 * 1000 / hr #60 seconds per minute, 1000ms per s / heart rate = ms per heartbeat
                        elif rr_int == 0 and "CardiacNumberOfImages" in dicom: #if the nominal interval is zero but there is a cardiac num of images and rep time
                            rep_time = dicom.RepetitionTime
                            num_imgs = dicom.CardiacNumberOfImages
                            rr_int = rep_time * num_imgs
                    elif "HeartRate" in dicom: #if the nominal interval is missing but there is a heart rate,
                        hr = dicom.HeartRate
                        rr_int = 60 * 1000 / hr #60 seconds per minute, 1000ms per s / heart rate = ms per heartbeat
                    elif "CardiacNumberOfImages" in dicom: #this is not always very accurate
                        rep_time = dicom.RepetitionTime
                        num_imgs = dicom.CardiacNumberOfImages
                        rr_int = rep_time * num_imgs
                    else:
                        rr_int = 0



                    # write entries into csv file
                    row = [os.path.join(cinefolder, dcm), dicom.TriggerTime, dicom.SliceLocation, rr_int]
                    voxelsize = dicom.PixelSpacing[0] * dicom.PixelSpacing[1] * dicom.SliceThickness #removed "spacingbetweenslices" as it is not always present

                    # get counts of each pixel
                    label = segmented.astype(np.int16)
                    counts = [np.count_nonzero(label == 300), np.count_nonzero(label == 200), np.count_nonzero(label == 100)]
                    row.extend([counts[0], counts[1], counts[2], voxelsize])
                    writer.writerow(row)
        

        #read in the data from the 2D csv
        segdata_2d = pandas.read_csv(os.path.join(output_dir, 'segmentations.csv'))
        #make new columns to help with organizing 3D dataframe
        segdata_2d['seriesnames'] = segdata_2d['file path'].apply(lambda x: os.path.basename(os.path.dirname(x))) #get the series name
        #no longer going to use the series name for assuming separate slice acquisitions, as some series have multiple slice locations in one directory, but still keeping because they might not use the same series name for 2 diff location acquisitions?
        segdata_2d['slicerounded'] = segdata_2d['slice location (mm)'].apply(lambda x: round(x, 2)) #round the slice location to 2 decimal places
        segdata_2d['lvbpvol'] = segdata_2d['LV blood pool (pixels)'] * segdata_2d['voxel size (mm^3/pixel)'] * 0.001 #calculate segmentation contribution to volume and convert to mL
        segdata_2d['rvbpvol'] = segdata_2d['RV blood pool (pixels)'] * segdata_2d['voxel size (mm^3/pixel)'] * 0.001
        segdata_2d['lvmyovol'] = segdata_2d['LV myocardium (pixels)'] * segdata_2d['voxel size (mm^3/pixel)'] * 0.001
        segdata_2d['series_slice_id'] = segdata_2d['seriesnames'] + "-" + segdata_2d['slicerounded'].astype(str)
        #drop repeated series_slice_ids (keep last)
        unique_series_slice = segdata_2d.drop_duplicates(subset=['series_slice_id'], keep='last')
        #sort by series name
        unique_series_slice = unique_series_slice.sort_values(by=['seriesnames'])
        #drop repeated slice locations (keep last) from above - keep a list of these series_slice combos, then subset based on it
        unique_slices = unique_series_slice.drop_duplicates(subset=['slicerounded'])
        unique_slices = unique_slices.sort_values(by=['slicerounded'])

        #KEEP THIS LIST OF SERIES_SLICE_IDS
        keep_slice_series = unique_slices['series_slice_id'].values.tolist()
        segdata_2d_trimmed = segdata_2d[segdata_2d['series_slice_id'].isin(keep_slice_series)]

        #can now use previous methods for phase number etc.
        #list of R-R intervals
        rrs = segdata_2d_trimmed['R-R (ms)'].values.tolist()
        rr = sum(rrs) / len(rrs)


        #list of slice locations
        slicelocations = unique_slices['slicerounded'].values.tolist()
        #number of phases
        num_phases = len(segdata_2d_trimmed[segdata_2d_trimmed['slicerounded'] == slicelocations[0]])
        #for each slice location, slice, sort by trigger time, then add a position (ie 1 - num_phases)
        segdata_2d_trimmed['phasenum'] = ''
        segdata_2d_trimmed = segdata_2d_trimmed.sort_values(by=['trigger time (ms)'])
        for loc in slicelocations:
            #get the indices of all entries at that slice - use index
            indices = segdata_2d_trimmed[segdata_2d_trimmed['slicerounded'] == loc].index.tolist()
            for phase_ind in range(0, num_phases):
                #assign relative phase number in the 'slicenum' column at "indices'"
                thisind = indices[phase_ind]
                segdata_2d_trimmed.at[thisind, 'phasenum'] = phase_ind + 1

        #initialize the volume lists
        lvbplist = []
        lvmyolist = []
        rvbplist = []
        #for each phase, add up the volumes at every slice
        for phase in range(0, num_phases):
            lvbplist.append(segdata_2d_trimmed.loc[segdata_2d_trimmed['phasenum'] == (phase + 1), 'lvbpvol'].sum())
            lvmyolist.append(segdata_2d_trimmed.loc[segdata_2d_trimmed['phasenum'] == (phase + 1), 'lvmyovol'].sum())
            rvbplist.append(segdata_2d_trimmed.loc[segdata_2d_trimmed['phasenum'] == (phase + 1), 'rvbpvol'].sum())

        phaselist = range(1, num_phases + 1)

        #this is all of the columns we need, now make the new dataframe with 3d summarized data
        data_3d = {'Phase': phaselist, 'LVBP volume (mL)': lvbplist, 'LV myo volume (mL)': lvmyolist, 'RVBP volume (mL)': rvbplist}
        segdata_3d = pandas.DataFrame(data_3d)
        #write 3d data to csv
        segdata_3d.to_csv(os.path.join(output_dir, 'segdata3d.csv'), index=False)

        # #5/2 generate the difference in volume between each phase, difference in time between each phase
        dt = rr/num_phases #calculate time difference between phases

        ldv = []
        rdv = []
        for phasediff in range(0, (num_phases-1)):
            l = lvbplist[phasediff]
            lplusone = lvbplist[phasediff+1]
            ldv.append(lplusone - l)
            r = rvbplist[phasediff]
            rplusone = rvbplist[phasediff+1]
            rdv.append(rplusone - r)

        #5/19 change after discussion with walter, making this this most negative dv/dt (ie min) so that it's maximum absolute during systole (ventricle getting smaller)
        peak_ldv = min(ldv)
        peak_rdv = min(rdv)
        peak_ldvdt = peak_ldv / dt
        peak_rdvdt = peak_rdv / dt


        #make another sheet with information about the case for simple compilation of results: ESV, EDV, EFs, mass
        #can eventually have this written into a sheet that summarizes all of the cases being analyzed in the run
        #or can just make another script later that goes through everything and puts the data into a sheet
        #case id, LVEDV, LVESV, LVEF, RVEDV, RVESV, RVEF, LV mass 
        lvedv = max(lvbplist)
        lvesv = min(lvbplist)
        lvef = (lvedv - lvesv) / lvedv
        rvedv = max(rvbplist)
        rvesv = min(rvbplist)
        rvef = (rvedv - rvesv)/ rvedv
        myodensity = 1.055 # (g/mL) - confirmed correct by Fogel 4/18/23. this is from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6461811/#:~:text=The%20clinically%20accepted%20value%20of,15%E2%80%9322%5D%2C%20cf.
        diastole_ind = lvbplist.index(lvedv)
        lvmass = lvmyolist[diastole_ind] * myodensity #lv mass is calculated from diastole, so using LV diastole phase location as data from this case show slightly different lv myo vols here
        #idea: we could use offset-ness of the indices to roughly estimate conduction delay between the ventricles? would also want to know HR so we can estimate the actual time
        lvsysind = lvbplist.index(lvesv)
        rvsysind = rvbplist.index(rvesv)
        lvrvdelay = rvsysind - lvsysind #the absolute value of the difference in time between min LV and min RV, in phase counts. rv - lv because expect rv delay
        abslvrvdelay = abs(lvrvdelay) #absolute value of delay (phase counts)
        abslvrvdelay_time = abslvrvdelay * dt #absolute value of delay (ms)
        # lv_strain #figure this out
        # rv_strain #figure this out
        lv_cf = (lvedv - lvesv) / lvmass #stroke volume / LV mass --> mL/g
        # lv_systhirdflow #add this later
        # rv_systhirdflow #add this later

        data_summary = {'TOF ID': os.path.basename(output_dir), 'LVEDV (mL)': lvedv, 'LVESV (mL)': lvesv, 
                        'LVEF': lvef, 'LV mass (g)': lvmass, 'RVEDV (mL)': rvedv, 'RVESV (mL)': rvesv, 'RVEF': rvef, 
                        'LV Contraction Fraction (mL/g)': lv_cf, 'LV-RV Delay (number of phases)': lvrvdelay, 
                        'Absolute LV-RV Delay (number of phases)': abslvrvdelay, 'Absolute LV-RV Delay (ms)': abslvrvdelay_time,
                        'Peak LV dV (mL/phase)': peak_ldv, 'Peak LV dV/dt (mL/ms)': peak_ldvdt,
                        'Peak RV dV (mL/phase)': peak_rdv, 'Peak RV dV/dt (mL/ms)': peak_rdvdt}
        summarydata = pandas.DataFrame(data_summary, index = [0])
        summarydata.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)



    else:
        for cine, file_name in cines:
            segmented = segment_nii(cine, model)
            nb.save(segmented, os.path.join(output_dir, os.path.splitext(os.path.basename(file_name))[0] + '_segmented.nii'))

# callee from curator.py
def segment(input_dir, output_dir, use_dicom, segment_all):
    logger.info("segmentation.segment")
    
    if (use_dicom):
        logger.info("       Segmenting DICOMs")
    else:
        logger.info("       Segmenting Niftis")

    # NOTE: this is for debugging only. The segmentor will assume all subdirectories are CINE series and will
    if segment_all:
        logger.info("           Segmenting all series in the input directory")
        # create a segmentation folder with the name of the subject - add the name
        foldername = os.path.basename(os.path.normpath(input_dir))
        output_path = os.path.join(os.path.dirname(output_dir), foldername)
        if not(os.path.exists(output_path)):
            os.makedirs(os.path.join(output_path))

        processSeries(input_dir, output_path, use_dicom)

    else:
        # changes made in 2023/07/25: Segmentor requires to always input parent folder of the subject directories.
        for subject in os.listdir(input_dir):

            # check if it's a valid classified directory, and its subdirectory contains "SAX_CINE"
            tocheck = os.path.join(input_dir, subject)
            if os.path.isdir(tocheck) and "SAX_CINE" in os.listdir(tocheck):

                logger.info("           Processing study: {}".format(subject))
                foldername = os.path.basename(subject)
                output_path = os.path.join(output_dir, foldername)
                if not(os.path.exists(output_path)):
                    os.makedirs(os.path.join(output_path))

                sax_cines = os.path.join(tocheck, "SAX_CINE")
                processSeries(sax_cines, output_path, use_dicom)