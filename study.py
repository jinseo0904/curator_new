class Study():
    def __init__(self,StudyInstanceUID):
        self.StudyInstanceUID = StudyInstanceUID
        self.cpt = None
        self.studyDate = None
        self.seriesList = [] # array of series objects
        self.patientName = ""
        self.studyIndex = 1
        self.accessionNumber = None
        self.patientID = None
    
    def countSeries(self):
        return len(self.seriesList)
                
    def countImages(self):
        number_of_images_in_study = 0
        for series in self.seriesList:
            number_of_images_in_study += series.countImages()
        return number_of_images_in_study
        
    def setPatientName(self,patientName):
        # Patch series description to allow csv, folder and filename usage.
        self.patientName = str(patientName)
        special_char = ',@_!#$%^&*()<>?/\|}{~:;[]. '
        for i in special_char:
            self.patientName = self.patientName.replace(i, '_')
        
    def printStudy(self):
            logger.info("Study: %s",self.StudyInstanceUID)
            logger.info("    accessionNumber: %s, %s Images",self.accessionNumber, self.countImages())
            for series in self.seriesList:
                    logger.info("    Series: %s",series.SeriesInstanceUID)
                    logger.info("        Series no.: %s, Acq. no.: %s",series.seriesNumber,series.acquisitionNumber)
                    logger.info("        Description: %s, %s Images",series.seriesDescription,series.countImages())
                    logger.info("        Jabba: %s, Series Process Type: %s",series.isValidJabbaSeries,series.processType)
                    if not (series.niiFile==None):
                        logger.info("        NiiFile: %s", series.niiFile)

class Series():
    def __init__(self,StudyInstanceUID,SeriesInstanceUID,acquisitionNumber):
        self.StudyInstanceUID = StudyInstanceUID
        self.SeriesInstanceUID = SeriesInstanceUID
        self.accessionNumber =""
        self.patientName = ""
        self.acquisitionNumber = acquisitionNumber
        self.seriesNumber=""
        self.seriesDescription=""
        self.dcmDirectory=""
        self.niiDirectory=""
        self.niiFile=""
        self.isValidSeries=""
        self.processType=""
        self.isContrastEnhanced=False
        self.imageList = [] # array of image objects

    def setDescription(self,seriesDescription):
        # Patch series description to allow csv, folder and filename usage.
        self.seriesDescription = str(seriesDescription)
        special_char = ',@_!#$%^&*()<>?/\|}{~:;[]. '
        for i in special_char:
            self.seriesDescription = self.seriesDescription.replace(i, '_')
            
    def setPatientName(self,patientName):
        # Patch series description to allow csv, folder and filename usage.
        self.patientName = str(patientName)
        special_char = ',@_!#$%^&*()<>?/\|}{~:;[]. '
        for i in special_char:
            self.patientName = self.patientName.replace(i, '_')
            
    def setNiiDirectory(self,path_to_nii):
        self.niiDirectory=path_to_nii
        
    def setNiiFile(self,niiFile):
        self.niiFile=niiFile
        
    def setDcmDirectory(self,path_to_series):
        self.dcmDirectory=path_to_series

    def countImages(self):
        return len(self.imageList)
        
    def __eq__(self,other):
        return (self.SeriesInstanceUID==other.SeriesInstanceUID)and(self.acquisitionNumber==other.acquisitionNumber)
        
class Image():
    def __init__(self,SOPInstanceUID):
        self.acquisitionNumber = None
        self.filename = None
        self.scantype = None
        self.SOPInstanceUID = SOPInstanceUID
        self.sliceLocation = None
        self.TriggerTime = None