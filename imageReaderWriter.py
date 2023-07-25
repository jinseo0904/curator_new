# system
import os
from os import walk
from tqdm import tqdm
import csv
import shutil
import logging
logger=logging.getLogger("curator")

# pydicom
import pydicom

# custom modules
from study import Study, Series, Image

class imageReaderWriter():
    def __init__(self,name):
        logger.info("imageReaderWriter.init()")
        self.name = name
        self.output_dir = None
        self.studyList = []
        
    def printStudyList(self):
        for study in self.studyList:
            study.printStudy()
            
    def getSeriesProcessType(self,meta):

        if not 'ProcedureCodeSequence' in meta:
            return ["NONE"]
        proc = str(meta.ProcedureCodeSequence[0].CodeValue)

        # Check for bolus info
        hasBolus = False
        #bolusTags = ["ContrastBolusStartTime", "ContrastBolusStopTime", "ContrastBolusRoute", "ContrastBolusAgent", "ContrastBolusVolume", "ContrastBolusTotalDose", "ContrastBolusFlowRate", "ContastBolusFlowDuration", "ContrastBolusIngredient", "ContrastBolusIngredientConcentration"]
        bolusTags = ["ContrastBolusStartTime",
            "ContrastBolusStopTime", 
            "ContrastBolusRoute", 
            "ContrastBolusAgent", 
            "ContrastBolusVolume", 
            "ContrastBolusTotalDose",
            "ContrastBolusIngredient", 
            "ContrastBolusIngredientConcentration"]
        for bTag in bolusTags:
            if bTag in meta:
                if str(getattr(meta, bTag)) != '':
                    hasBolus = True

        fatCodes = [ "CTABEZ", "CTABCZ", "CTAPCZ", "CTAPEZ", "CTAPUZ" ]
        liverCodes = [ "CTABCZ", "CTAPCZ", "CTAPUZ", "CTCAPU3DZ", "CTCHCAZ", "CTCHCZ", "CTCHULPZ", "CTCHUZ"]

        if proc in fatCodes:
            if proc in liverCodes:
                if hasBolus:
                    return ["FAT"]
                else:
                    return ["FAT", "LIVER"]
            else:
                return ["FAT"]
        else:
            if proc in liverCodes:
                if not hasBolus:
                    return ["LIVER"]

        return ["NONE"]
        
    def copyValidSeriesAndUpdateStudyList(self,copy_dir,use_patientname_as_foldername=False,use_cmr_info_as_filename=False,copy_all_series=True):
        logger.info("FileReader.copyValidSeriesAndUpdateStudyList()")
        logger.info("    Copying valid series to %s",copy_dir)
        logger.info("    use_cmr_info_as_filename %s",use_cmr_info_as_filename)
        logger.info("    use_patientname_as_foldername %s",use_patientname_as_foldername)
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir,exist_ok=True)
        # pass through the study list, copy valid files, and update path
        for studyIndex, study in tqdm(enumerate(self.studyList[:])):
            for seriesIndex, series in enumerate(study.seriesList[:]):
                if(series.isValidSeries or copy_all_series):
                    if(use_patientname_as_foldername==True):
                        logger.info("    writing study folder: %s",study.patientName)
                        studyDirectory=os.path.join(copy_dir,study.patientName)
                    else:
                        studyDirectory=os.path.join(copy_dir,study.accessionNumber)
                        logger.info("    writing study folder: %s",study.accessionNumber)
                    seriesDirectory=os.path.join(studyDirectory,str(series.seriesNumber)+"_"+series.seriesDescription)
                    if not os.path.exists(studyDirectory):
                        os.makedirs(studyDirectory,exist_ok=True)
                    if not os.path.exists(seriesDirectory):
                        os.makedirs(seriesDirectory,exist_ok=True)
                    for imageIndex, image in enumerate(series.imageList[:]):
                        if(use_cmr_info_as_filename==True):
                            new_filename = os.path.join(seriesDirectory,str(image.acquisitionNumber)+"_"+str(int(image.sliceLocation+1000))+"_"+str(int(image.TriggerTime))+"_"+image.SOPInstanceUID+'.dcm')
                        else:
                            new_filename = os.path.join(seriesDirectory,str(image.acquisitionNumber)+"_"+image.SOPInstanceUID+'.dcm')
                        shutil.copyfile(image.filename,new_filename)
        self.studyList = []
        self.studyList=self.queryDirectory(copy_dir,level=2)
        return self.studyList
        
    def queryDirectory(self,input_dir,level=1):
        logger.info("FileReader.queryDirectory()")
        logger.info("    Querying Directory for Dicom Files")
        logger.info("    Searching %s for dicom files",input_dir)
        logger.info("    Searching (at least) %s folder level(s) deep",level)
        # retrieve a list of all files at the directory level specified
        if(level==1):
            filelist = os.listdir(input_dir)
        elif(level>1):
            filelist=[]
            for (dirpath,dirnames,filenames) in walk(input_dir):
                for x in filenames:
                    filelist.append(os.path.join(dirpath,x))
        else:
            logger.info("    directory search level %s is not valid",level)
        for filename in tqdm(filelist):
            self.addFileToStudyList(filename)
        return self.studyList
    
    def addFileToStudyList(self,filename):
        try:
            meta = pydicom.dcmread(filename,stop_before_pixels=True)
        except:
            return
        try:
            tag = "StudyInstanceUID"
            StudyInstanceUID = meta.data_element(tag).value
        except:
            return
        try:
            tag = "SeriesInstanceUID"
            SeriesInstanceUID = meta.data_element(tag).value
        except:
            logger.error("Invalid SeriesInstanceUID in file: " + filename + "\n")
            return
        try:
            tag = "SOPInstanceUID"
            SOPInstanceUID = meta.data_element(tag).value
        except:
            logger.error("Invalid SOPInstanceUID in file: " + filename + "\n")
            return
        try:
            tag = "AcquisitionNumber"
            acquisitionNumber = meta.data_element(tag).value
        except:
            acquisitionNumber=None
            logger.error("Invalid acquisitionNumber in file: " + filename + "\n")
        try:
            tag = "AccessionNumber"
            accessionNumber = meta.data_element(tag).value
        except:
            logger.error("Invalid accessionNumber in file: " + filename + "\n")
            return
        try:
            tag = "PatientName"
            patientName = str(meta.data_element(tag).value)
        except:
            patientName=""
            return
        try:
            tag = "SliceLocation"
            sliceLocation = meta.data_element(tag).value
        except:
            sliceLocation=""
            return
        try:
            tag = "TriggerTime"
            TriggerTime = meta.data_element(tag).value
        except:
            TriggerTime=""
            return

        studyfound = False
        for study in self.studyList:
            if(study.StudyInstanceUID==StudyInstanceUID):
                studyfound = True
        if not studyfound: # add study to study list
            newstudy = Study(StudyInstanceUID)
            newstudy.accessionNumber=accessionNumber
            newstudy.setPatientName(patientName)
            self.studyList.append(newstudy)

        # which study does this file belong to?
        for j in range(0,len(self.studyList)):
            current_study = self.studyList[j]
            current_uid = current_study.StudyInstanceUID
            if(current_uid == StudyInstanceUID):
                studyIndex = j

        seriesFound=False

        for series in self.studyList[studyIndex].seriesList:
            if(SeriesInstanceUID==series.SeriesInstanceUID)and(acquisitionNumber==series.acquisitionNumber):
                seriesFound=True
        
        if not seriesFound:
            newSeries=Series(StudyInstanceUID,SeriesInstanceUID,str(acquisitionNumber))
            newSeries.setDcmDirectory(os.path.dirname(filename))
            newSeries.setPatientName(patientName)
            newSeries.accessionNumber=accessionNumber
            if(self.validateSeries(meta)):
                newSeries.isValidSeries=True
                seriesProcessType=self.getSeriesProcessType(meta)
                newSeries.processType=str(seriesProcessType)
            try:
                tag = "SeriesDescription"
                newSeries.setDescription(meta.data_element(tag).value)
            except:
                logger.error("Not a valid series description in file: %s",filename)
            try:
                tag = "SeriesNumber"
                newSeries.seriesNumber=str(meta.data_element(tag).value+10000)
            except:
                logger.error("Not a valid series number in file: %s",filename)
            newSeries.accessionNumber=str(accessionNumber)
            self.studyList[studyIndex].seriesList.append(newSeries)
            #logger.info("    Series found: %s",SeriesInstanceUID)

        # which series does this file belong to
        for j in range(0,len(self.studyList[studyIndex].seriesList)):
            current_series_uid = self.studyList[studyIndex].seriesList[j].SeriesInstanceUID
            if(current_series_uid==SeriesInstanceUID):
                seriesIndex = j

        # add the file to the file list for that series
        newimage = Image(SOPInstanceUID)
        newimage.acquisitionNumber = acquisitionNumber
        newimage.filename = filename
        newimage.sliceLocation = sliceLocation
        newimage.TriggerTime = TriggerTime
        self.studyList[studyIndex].seriesList[seriesIndex].imageList.append(newimage)

    # Check if a series is suitable for processing
    def validateSeries(self,meta):
        try:
            if meta.ImageType[0] != "ORIGINAL":
                #logger.info( "========> ORIGINAL images only")
                return False
        except:
            return False

        return True
        
    def writeStudyListToCSV(self,csv_file,csv_overwrite=True):
        logger.info("writeStudyListToCSV()")
        if not self.studyList:
            logger.info("    No studies in studylist. No csv written.")
            return
        if os.path.isfile(csv_file) and csv_overwrite==False:
            logger.info("    csv overwrite = %s, Not writing csv: %s",csv_overwrite,csv_file)
            return
        with open(csv_file,"w") as f:
            writer=csv.writer(f)
            row=[]
            for attr, value in Series("","","").__dict__.items():
                if(isinstance(value, str)):
                    row.append(attr)
            writer.writerow(row)
            for study in self.studyList:
                for series in study.seriesList:
                    row=[]
                    for attr, value in series.__dict__.items():
                        if(isinstance(value, str)):
                            row.append(value)
                    writer.writerow(row)
        logger.info("    csv file written: %s",csv_file)
                

    def convertDicomToNifti(self,nii_dir,nii_overwrite=False):
        logger.info("convertDicomToNifti(nii_dir)")
        if not self.studyList:
            logger.info("List of studies to convert is empty")
            return
        for study in self.studyList:
            try:
                studyDirectory=os.path.join(nii_dir,study.accessionNumber)
            except:
                studyDirectory=os.path.join(nii_dir.study.StudyInstanceUID)
                logger.info("Could not set studyDirectory using accession: %s",study.accessionNumber)
            for series in study.seriesList:
                niiFilename=os.path.join(studyDirectory,str(series.seriesNumber)+"_"+series.seriesDescription+".nii")
                series.setNiiDirectory(studyDirectory)
                series.setNiiFile(niiFilename)
                if not os.path.exists(studyDirectory):
                    os.makedirs(studyDirectory,exist_ok=True)
                if os.path.isfile(niiFilename) and nii_overwrite==False:
                    logger.info("    nii overwrite = %s, Skipping: %s",nii_overwrite,niiFilename)
                    continue
                if(series.dcmDirectory==None):
                    logger.info("No valid directory for series: %s",series.SeriesInstanceUID)
                    continue
                try:
                    logger.info("     Converting dicom series in folder: %s",series.SeriesInstanceUID)
                    result=dicom2nifti.dicom_series_to_nifti(series.dcmDirectory,niiFilename) #reorient?
                except ConversionError as err:
                    logger.error(err)
                    continue
                except ConversionValidationError as err:
                    logger.error(err)
                    continue
                except:
                    logger.error("    Unknown error during dicom to nii conversion")
                    logger.error("    Series: %s",series.SeriesInstanceUID)
                    logger.error("    niiFile: %s",niiFilename)
                    continue
                if not os.path.isfile(niiFilename):
                    logger.error("  >>> ERROR - nifti conversion failed")
                    logger.error("Series: %s",series.SeriesInstanceUID)
                    logger.error("niiFile: %s",niiFilename)
                    continue


# class FileReaderCSV(FileReader):
    # def __init__(self,csv_file):
        # super().__init__("FileReaderCSV")
        # self.csv_file = csv_file
    
# class FileReaderDir(FileReader):
    # def __init__(self,input_dir,output_dir):
        # super().__init__("FileReaderDir")
        # self.input_dir = input_dir
        # self.output_dir = output_dir