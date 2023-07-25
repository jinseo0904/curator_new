# system
import argparse
import os
import sys
import logging

# other dependencies
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

orientation_names = ['2CH','3CH','4CH','AOFLOW','APEX','AXIAL','AXIAL_MIP','BASE','CORONAL_MIP','DAO','IVC','LOC',
                     'LPA','LPV','LVOT','MID','MPA','RPA','RPV','RV2CH','RVOT','SAGITTAL_MIP','SAG_RVOT','SAX','SVC','TRICUSPID']

contrast_dict = {'CINE': 0, 'DE': 1, 'HASTE': 2, 'PC_MAG': 3, 'PC_PHASE': 4, 'PERF_AIF': 5, 'PSIR_MAG': 6, 'PSIR_PHASE': 7,
                  'SCOUT': 8, 'T1': 9, 'T1*': 10, 'T1RHO': 11, 'TRUFI': 12, 'TWIST': 13}
contrast_names = list(contrast_dict.keys())

orientation_dict = dict(zip(orientation_names, range(len(orientation_names))))


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

def make_comparison(prediction_path, ground_truth_path, output_dir):
    # load and process ground truth csv file
    try:
        df = pd.read_csv(ground_truth_path)
    except:
        logger.error("    Failed loading the Ground Truth csv file. Please check your input path again.")
        sys.exit()

    filenames = [patient + "_" + str(num) + "_" + series for (patient, num, series) in zip(df.patientName, df.seriesNumber, df.seriesDescription)]
    df["File Name"] = filenames
    gt = df.drop_duplicates(subset=['File Name'], keep='first')

    # load DL model predictions
    try:
        df = pd.read_csv(prediction_path)
    except:
        logger.error("    Failed loading the DL predictions csv file. Please check your input path again.")

    pred = df[["Subject Name", "File Name", "Orientation", "Contrast"]]
    patient_label = [patient + "_" + name for (patient, name) in zip(df["Subject Name"], df["File Name"])]
    pred["fullSeriesPath"] = patient_label

    # check if each file name exists in the classification csv file
    exists = [contains(s, pred["fullSeriesPath"]) for s in gt["File Name"]]
    gt["Exists"] = exists

    # reorganize columns and capitalize orientation/contrast labels
    checked = gt[["File Name", "Exists", "Orientation", "Contrast"]]
    checked["GT_Ori"] = [str.upper(x) for x in checked["Orientation"]]
    checked["GT_Con"] = [str.upper(x) for x in checked["Contrast"]]

    # fetch predictions
    available = checked[checked["Exists"] == True]
    gt_ori = []
    gt_con = []
    subject_names = []    

    for name in available["File Name"]:
        result = pred[pred['fullSeriesPath'] == name]
        gt_ori.append(result.at[result.index[0], 'Orientation'])
        gt_con.append(result.at[result.index[0], 'Contrast'])
        subject_names.append(result.at[result.index[0], 'Subject Name'])

    available["Subject Name"] = subject_names
    available["Orientation"] = gt_ori
    available["Contrast"] = gt_con
    available = available[["Subject Name", "File Name", "Exists", "Orientation", "Contrast", "GT_Ori", "GT_Con"]]
    
    sorted_df = available.sort_values(by='File Name')

    # create checks column
    sorted_df["Correct_Ori"] = sorted_df["Orientation"] == sorted_df["GT_Ori"]
    sorted_df["Correct_Con"] = sorted_df["Contrast"] == sorted_df["GT_Con"]

    # write to csv
    
    filename = 'comparisons_' + str(datetime.now().month)
    filename += "_" + str(datetime.now().day)
    filename += "_" + str(datetime.now().hour)
    filename += "_" + str(datetime.now().minute) + ".csv"
    sorted_df.to_csv(os.path.join(output_dir, filename), index=False)

    # log accuracies
    or_hits = sorted_df['Correct_Ori'].value_counts()[True]
    or_fails = sorted_df['Correct_Ori'].value_counts()[False]

    con_hits = sorted_df['Correct_Con'].value_counts()[True]
    con_fails = sorted_df['Correct_Con'].value_counts()[False]

    logger.info("Orientation Accuracy: {}".format(or_hits / (or_hits + or_fails)))
    logger.info("Contrast Accuracy: {}".format(con_hits / (con_hits + con_fails)))

    # make confusion matrix
    sns.color_palette("light:b", as_cmap=True)
    cm = confusion_matrix(sorted_df["GT_Ori"], sorted_df["Orientation"], labels=orientation_names)
    df_cm = pd.DataFrame(cm, orientation_names, orientation_names)

    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    sns.heatmap(df_cm, annot=True, fmt='g', ax=axs[0], cmap="Blues", vmax=40)  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    axs[0].set_xlabel('Predicted labels')
    axs[0].set_ylabel('True labels')
    axs[0].set_title('Orientation Confusion Matrix') 

    cm2 = confusion_matrix(sorted_df["GT_Con"], sorted_df["Contrast"], labels=contrast_names)
    df_cm2 = pd.DataFrame(cm2, contrast_names, contrast_names)
    sns.heatmap(df_cm2, annot=True, fmt='g', ax=axs[1], cmap="Blues", vmax=40)
    axs[1].set_xlabel('Predicted labels')
    axs[1].set_ylabel('True labels')
    axs[1].set_title('Contrast Confusion Matrix') 

    #plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)

    return

def contains(a, b):
    for compare in b:
        if a == compare:
            return True
    return False


# main method
if __name__ == "__main__":
    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compares curator classifier output with ground truth labels")
    parser.add_argument("--predictions", type=str, help="Path of classifications csv file")
    parser.add_argument("--ground_truth", type=str, help="Path of ground truth csv/Excel file")
    parser.add_argument("--output_dir",type=str,help="Output directory with curated files")
    parser.add_argument("--log_file",type=str,help="txt file with log info",default="curator_log.txt")
    args = parser.parse_args()
    
    # Initialize logger
    logger=initLogger("curator",args.log_file)

    if args.predictions is None or args.ground_truth is None or args.output_dir is None:
            logger.error("    Check that --predictions, --ground_truth, --output_dir are defined")
            sys.exit()

    make_comparison(args.predictions, args.ground_truth, args.output_dir)


