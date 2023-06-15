#!/usr/bin/env python

""" Script to find duplicated images in a target path """

# common
import os
import sys
import re
import logging
import pickle
import importlib

# opencv
import cv2

# configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def remove_thumbs(path):
    """ Remove thumb files from path"""
    files = os.listdir(path)
    for filename in files:
        if "thumb" in filename:
            os.remove(os.path.join(path, filename))

def main():
    """ Start the script """
    config = importlib.import_module('config')

    logging.info("Start with cv2 version: %s", cv2.__version__)

    # remove thumbs
    remove_thumbs(config.TARGET_PATH)
    remove_thumbs(config.SEARCH_PATH)

    # Load processed files and their features
    feat_file = "features.pkl"
    processed_files, target_index = [], []
    if os.path.exists(feat_file):
        logging.info("Pickle file %s found, load existent object structure", feat_file)
        with open(feat_file, 'rb') as feat:
            try:
                processed_files, target_index = pickle.load(feat)
            except EOFError as ex:
                logging.debug("Pickle file %s found but damaged (%s), start new", feat_file, str(ex))

    for filename in os.listdir(config.TARGET_PATH):
        file_path = os.path.join(config.TARGET_PATH, filename)
        if not filename in processed_files:
            image_target = cv2.imread(file_path)

            # If image contains '@', scale it and rename it.
            if "@" in filename:
                image_number = re.search(r"photo_(\d+)@\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}.jpg", filename).group(1)
                match = re.search(r"\d{2}-\d{2}-\d{4}", filename)
                if match:
                    date = match.group(0)
                else:
                    continue

                if image_target.shape != (500, 500, 3):
                    image_target = cv2.resize(image_target, (500, 500))
                    # save scaled image
                    new_name = os.path.join(config.TARGET_PATH, f"photo_{image_number}_{date}.jpg")
                    logging.debug("Scale image %s", new_name)
                    if cv2.imwrite(new_name, image_target):
                        # remove original
                        logging.debug("Remove old image: %s", file_path)
                        os.remove(file_path)
                    file_path = new_name

            orb = cv2.ORB_create()
            _kp, des = orb.detectAndCompute(image_target, None)
            target_index.append((file_path, des))
            processed_files.append(filename)

            # Save processed files and their features
            with open(feat_file, 'wb') as feat:
                pickle.dump((processed_files, target_index), feat)

    # Start comparison
    logging.info("Starting image comparison...")
    for _idx, filename in enumerate(os.listdir(config.SEARCH_PATH), 1):
        # search only images with date as CHECK_DATE
        date = re.search(r"\d{2}-\d{2}-\d{4}", filename).group(0)
        if config.CHECK_DATE != date:
            continue

        file_path = os.path.join(config.SEARCH_PATH, filename)
        search_image = cv2.imread(file_path)

        if search_image.shape != (500, 500, 3):
            search_image = cv2.resize(search_image, (500, 500))

        orb = cv2.ORB_create()
        _kp, search_des = orb.detectAndCompute(search_image, None)

        duplicate_found = False  # Track whether a duplicate has been found

        for target_filename, target_des in target_index:
            bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_match.match(search_des, target_des)
            if len(matches) > 300:
                # extract image number and date from target_filename
                match = re.search(r"photo_(\d+)_(\d{2}-\d{2}-\d{4}).jpg", target_filename)
                if match:
                    image_number = match.group(1)
                    date = match.group(2)
                    # convert the date format
                    date = date.replace("-", ".")
                    # construct and print the message
                    message = f"photo-{date}"
                    logging.info(f"Possible match found for {filename} and {message} (score: {len(matches)})")
                    duplicate_found = True  # Set the flag to True as a duplicate has been found
                    break

        # If no duplicate has been found after checking all target images, log the message
        if not duplicate_found:
            logging.info("No duplicate found for %s in target path", filename)

    logging.info("Script finished!")

if __name__ == "__main__":
    main()
