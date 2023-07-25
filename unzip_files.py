import os, zipfile
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    # command line parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unzipper - Unzip all files in a directory")
    parser.add_argument("input_dir", type=str, help="Input directory with files to be unzipped")
    args = parser.parse_args()

    dir_name = args.input_dir
    extension = ".zip"

    os.chdir(dir_name) # change directory from working dir to dir with files

    for item in tqdm(os.listdir(dir_name)): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name) # extract file to dir
            zip_ref.close() # close file
            #os.remove(file_name) # delete zipped file
