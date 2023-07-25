# system
import os
import subprocess
import logging
import glob
from tqdm import tqdm
logger=logging.getLogger("curator")

# dicom2nifti
import dicom2nifti

# pandas
import pandas as pd
import json

# pydicom
import pydicom

class niftiEngine():

    def __init__(self,name):
        logger.info("niftiEngine.init()")
        self.name = name
    
    def writeCSVToNifti(self,csv_file,output_dir,include_tags_txt,use_patientname_as_foldername=False):
        logger.info("niftiEngine.convertCSVToNifti")
        df=pd.read_csv(csv_file)
        
        # error checking
        if not set(["dcmDirectory","patientName","accessionNumber","acquisitionNumber","seriesNumber","seriesDescription"]).issubset(df.columns):
            logger.error("niftiEngine could not find column dcmdir in csv file")
            return
        
        with open(include_tags_txt, 'r') as f:
            include_tags = [l.strip() for l in f.readlines()]
        
        for index, row in tqdm(df.iterrows()):
            dcmDirectory = row["dcmDirectory"]
            seriesNumber = row["seriesNumber"]
            seriesDescription = row["seriesDescription"]
            accessionNumber = row["accessionNumber"]
            patientName = row["patientName"]
            #contrast = row["contrast"]
            #orientation = row["orientation"]
            nii_output_dir=os.path.join(output_dir,str(accessionNumber))
            logger.info("Writing " + nii_output_dir)
            self.process_dcm(dcmDirectory,nii_output_dir,include_tags)

        for index, row in tqdm(df.iterrows()):
            dcmDirectory = row["dcmDirectory"]
            seriesNumber = row["seriesNumber"]
            seriesDescription = row["seriesDescription"]
            accessionNumber = row["accessionNumber"]
            patientName = row["patientName"]
            #contrast = row["contrast"]
            #orientation = row["orientation"]

            # checks for --use_patientname_as_foldername argument
            if(use_patientname_as_foldername==True):
                nii_output_dir=os.path.join(output_dir,patientName)
            else:
                nii_output_dir=os.path.join(output_dir,str(accessionNumber))
            logger.info("Writing " + nii_output_dir)
            self.process_dcm(dcmDirectory,nii_output_dir,include_tags)
    
    def process_tag_dict(self,raw_tag_dict, unpack_singleton_values=False):
        # input: tag dict conforming to the DICOM JSON model
        # (see http://dicom.nema.org/dicom/2013/output/chtml/part18/sect_F.2.html)
        # output: human-readable tag dict
        processed_tag_dict = {}
        for tag_id, tag_body in raw_tag_dict.items():
            try:
                tag_keyword = pydicom.datadict.dictionary_keyword(tag_id)
                if 'Value' in tag_body:
                    tag_vr = tag_body['vr']
                    tag_values = tag_body['Value']
                    assert isinstance(tag_values, list)
                    # recursively process sequences
                    if tag_vr == 'SQ':
                        for i in range(len(tag_values)):
                            tag_values[i] = process_tag_dict(
                                tag_values[i], unpack_singleton_values=unpack_singleton_values)
                    if unpack_singleton_values and len(tag_values) == 1:
                        tag_values = tag_values[0]
                    processed_tag_dict[tag_keyword] = tag_values
            except:
                logging.warning('      dictionary_keyword(%s) not found',tag_id)
                #logging.error("Exception occurred")

        return processed_tag_dict


    def process_dcm(self,input_dir, output_dir,
                    include_tags=[], inject_tags={}, write_slice_jsons=False):
        

        os.makedirs(output_dir, exist_ok=True)
        # output filenames
        slice_metadata_file = "slice_metadata.json"
        scan_metadata_file = "scan_metadata.json"
        log_file = "dcm2niix_log.txt"
        cmd_file = "dcm2niix_cmd.txt"
        
        # convert DCM to NII
        #if os.path.exists(output_dir):
        #   os.remove(output_dir)
        logging.info('      dcm2niix()')
        try:
            with open(os.path.join(output_dir, log_file), 'w') as f_log:
                result = subprocess.run(
                    ['dcm2niix', '-f', '%s_%d', '-m', 'y', '-o', output_dir, input_dir],
                    stdout=f_log, stderr=subprocess.STDOUT, universal_newlines=True)
        except:
            logging.exception("Exception occurred")
        with open(os.path.join(output_dir, cmd_file), 'w') as f_cmd:
            f_cmd.write(' '.join(result.args))
            logging.info("      "+' '.join(result.args))
        #if result.returncode != 0:
        #    print(f'{output_dir}: dcm2niix failed with code {result.returncode}')
        # extract metadata
        dcm_paths = glob.glob(os.path.join(input_dir, '*.dcm'))
        slice_tag_dicts = []
        if not write_slice_jsons:
            dcm_paths = dcm_paths[0:1] # only process first slice
        for dcm_path in dcm_paths:
            logging.info('      dcmread()')
            try:
                dcm_obj = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            except:
                logging.exception("Exception occurred")
            logging.info('      process_tag_dict()')
            try:
                raw_tag_dict = self.process_tag_dict(dcm_obj.to_json_dict(), unpack_singleton_values=True)
            except:
                logging.exception("Exception occurred")
            processed_tag_dict = {}
            processed_tag_dict["DicomDirectory"] = input_dir
            processed_tag_dict["DicomSliceFile"] = dcm_path
            processed_tag_dict["NiftiVolumeFile"] = output_dir if os.path.exists(output_dir) else None
            for tag in include_tags:
                processed_tag_dict[tag] = raw_tag_dict.get(tag)
            processed_tag_dict.update(inject_tags)
            slice_tag_dicts.append(processed_tag_dict)
        # TODO - remove tags that are different across slices?
        logging.info('      dcm metadata dump()')
        try:
            with open(os.path.join(output_dir, scan_metadata_file), 'w') as f:
                json.dump(slice_tag_dicts[0], f, indent=4)
        except:
            logging.exception("Exception occurred: json.dump")
        if write_slice_jsons:
            with open(os.path.join(output_dir, slice_metadata_file), 'w') as f:
                json.dump(slice_tag_dicts, f, indent=4)