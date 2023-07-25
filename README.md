# Curator - Bespoke Medical Image Curation

## Setup

Basic overview of the installation steps.

We have the choice of using Anaconda in native Win 11 or in Ubuntu 20.04 if using the Windows Subsystem for Linux

(Optional) Install [WSL2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Install [Anaconda](https://www.anaconda.com/) (Download for Linux on WSL2 or Windows for Win11)

Create a new conda environment
```
conda create --name curator python=3.8.12
```

Activate the new environment
```
conda activate curator
```

This has been tested on Python 3.8.12

Make sure the following package versions are installed (via pip):
```
dicom2nifti>=2.3.0
pydicom>=2.2.2
tqdm>=4.62.3
pandas>=1.3.4
torch >= 1.13.1
monai
cv2
```

## Additional CUDA setup notes

```
```

## Usage

```
python curator.py [-h] mode

python curator.py parse --input_dir /dcmdirectory/ --output_dir /outdcmdirectory/ --csv_file csv_file.csv

python curator.py nifti --csv_file csv_file.csv --output_dir /outdcmdirectory/

python curator.py classify --input_dir /dcmdirectory/ --output_dir /classified_directory/

python curator.py segment --input_dir /classified_dcm_directory/ --output_dir /classified_directory/

```

## Arguments

* `mode`: curator mode: ['parse','nifti','train','inference','segment']
* `--input_dir`: folder containing dicom images to be curated (will search all subdirectories of this folder).
* `--output_dir`: folder to place all the sorted dicom images by accession and series
* `--csv_file`: csv file containing a summary of all studies and series. Each series is a row.
* `--log_file`: Log file where results will be stored
* `--use_patientname_as_foldername`: (Parse) optional feature to use dicom field PatientName as foldername (default is AccessionNumber).
* `--use_cmr_info_as_filename`: (Parse) optional feature to place dicom fields SlicePosition_TriggerTime in dicom filename
* `--sax_cine_only`: (Classify) the classifier will only copy paste sax cine images to the output directory

# Additional usage details 

* mode = `parse`
    * Read a folder containing unlabeled dicom images. sort and copy the files, and generate a csv file summarizing all the valid series for labeling.
    * Mandatory arguments: --input_dir, --output_dir, --csv_file


* mode = `nifti`
    * Read a csv file containing columns "dcmdir" and "label". Images in dcmdir will be converted to nifti, one nifti per row.
    * Hint: Run parse to generate a csv file with dcmdir, add labels
    * Mandatory arguments: --csv_file, --output_dir


* mode = `train`
    * Train a NN to label imaging data
    * Mandatory arguments: --input_dir, --csv_file


* mode = `classify`
    * Given a trained NN, make new inferences on unseen data
    * Mandartory arguments: --input_dir, --output_dir


* mode = `segment`
    * Given a trained NN, make new inferences on unseen data
    * Mandartory arguments: --input_dir, --output_dir

### Additional notes on using the classifier and segmentor
When executing the curator in `classify` mode, your input directory should resemble the following structure:

    Studies_folder              # The --input_dir argument should include the path to this folder.
    ├── Subject1                    # subdirectory representing a single study
    │   ├── series1
    │   │   ├── 001.dcm
    │   │   ├── 002.dcm
    │   │   ├── 003.dcm             
    │   │   └── ...             
    │   ├── series2       
    │   └── ...          
    │ 
    │       
    ├── Subject2
    ├── Subject3
    └── ...

After the classifier finishes running, it will generate new folders in the output directory, sorting series according to orientation and contrast
    
    output_directory                # The specified output directory
    ├── dcm_date_and_time           
    │   ├── Classified_Subject1     # Series are sorted according to orientation and contrast
    │   │   ├── SAX_CINE
    │   │   ├── LOC_TRUFI 
    │   │   └── ...      
    │   │
    │   ├── Classified_subject2       
    │   └── ...          
    │ 
    └── ...
When utilizing the segmentor, please ensure that your input directory consists of the folder containing all subjects: `dcm_date_and_time`.
The segmentor will retrieve all series from the "SAX_CINE" folder within each study folder and segment them.