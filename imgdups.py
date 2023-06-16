#!/usr/bin/env python

""" Script to find duplicated images in a target path """

# common
import os
import sys
import logging
import pickle
import importlib
import numpy as np

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

def load_config():
    """Load the config.py as module"""
    config = importlib.import_module('config')
    if ( not os.path.exists(config.TARGET_PATH)
            or not os.path.exists(config.SEARCH_PATH) ):
        logging.error("Folder TARGET_PATH or SEARCH_PATH is "
                "missing, check your config.py. Exit!")
        sys.exit(1)

    return config

def remove_thumbs(path):
    """Remove thumb files from path"""
    files = os.listdir(path)
    for filename in files:
        if "thumb" in filename:
            os.remove(os.path.join(path, filename))

def load_pickle(file):
    """Load processed files and their features"""
    index = []

    if os.path.exists(file):
        logging.info("Pickle file %s found, load existent object structure", file)
        with open(file, 'rb') as feat:
            try:
                _processed_files, index = pickle.load(feat)
            except EOFError as ex:
                logging.debug("Pickle file %s found but damaged (%s), reset!",
                        file, str(ex))
                os.remove(file)

    return index

def clean_up_pickle(path, index):
    """clean up pickle data"""
    clean_index = []
    clean_processed_files = []
    for file, des in index:
        original_path = os.path.join(path, os.path.basename(file))
        if ( file not in clean_index
                and os.path.exists(original_path) ):
            clean_index.append((file, des))
            clean_processed_files.append(os.path.basename(file))

    return clean_processed_files, clean_index

def check_scaled_images(path, processed_files):
    """only keep processed files as scaled images"""
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # check if only processed files exist in scaled image folder
        scaled_images = [f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))]
        for image in scaled_images:
            if not image in processed_files:
                # remove old scaled image
                logging.debug("Remove obsolete scaled image: %s", image)
                os.remove(os.path.join(path, image))

def get_pickle_folder(path):
    """check pickle folder existence"""
    pickle_folder = os.path.join(path, "db")
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    return pickle_folder

def get_pickle(path):
    """Create a pickle file from target path"""
    scaled_image_folder = os.path.join(path, "imgdups")

    pickle_file = os.path.join(get_pickle_folder(scaled_image_folder), "image_cache.pkl")
    processed_files, index = clean_up_pickle(path, load_pickle(pickle_file))

    check_scaled_images(scaled_image_folder, processed_files)

    index_check = False

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        image_path = os.path.join(path, file)
        if not file in processed_files:
            image_target = cv2.imread(image_path)
            if image_target is None:
                continue

            if np.shape(image_target) != (500, 500, 1):
                image_target = cv2.resize(image_target, (500, 500))
                # save scaled image
                scaled_image_path = os.path.join(scaled_image_folder, file)
                if cv2.imwrite(scaled_image_path, image_target):
                    logging.debug("Add scaled image: %s", file)

            orb = cv2.ORB_create()
            _kp, des = orb.detectAndCompute(image_target, None)
            index.append((scaled_image_path, des))
            processed_files.append(file)
            index_check = True

    if index_check:
        # Save processed files and their features
        with open(pickle_file, 'wb') as feat:
            pickle.dump((processed_files, index), feat)

    return index

def main():
    """ Start the script """
    logging.info("Start script")
    exit_status = 0

    config = load_config()

    # remove thumbs
    remove_thumbs(config.TARGET_PATH)
    remove_thumbs(config.SEARCH_PATH)

    target_index = get_pickle(config.TARGET_PATH)

    # Start comparison
    logging.info("Starting image comparison...")

    files = [f for f in os.listdir(config.SEARCH_PATH)
            if os.path.isfile(os.path.join(config.SEARCH_PATH, f))]
    for filename in files:
        file_path = os.path.join(config.SEARCH_PATH, filename)
        search_image = cv2.imread(file_path)

        if np.shape(search_image) != (500, 500, 1):
            search_image = cv2.resize(search_image, (500, 500))

        orb = cv2.ORB_create()
        _kp, search_des = orb.detectAndCompute(search_image, None)

        duplicate_found = False

        for target_filepath, target_des in target_index:
            if os.path.basename(target_filepath) == "imgdups":
                continue
            bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_match.match(search_des, target_des)
            if len(matches) > 300:
                logging.info("%s == %s (score: %d)",
                        filename,
                        os.path.basename(target_filepath),
                        len(matches))
                duplicate_found = True
                exit_status += 1
                break

        # If no duplicate has been found after checking all target images, log the message
        if not duplicate_found:
            logging.info("No duplicate found for %s", filename)

    logging.info("Script finished!")
    if exit_status:
        sys.exit(exit_status)

if __name__ == "__main__":
    main()
