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
        logging.info("Cache file %s found, load existent object structure", file)
        with open(file, 'rb') as feat:
            try:
                _processed_files, index = pickle.load(feat)
            except EOFError as ex:
                logging.debug("Cache file %s found but damaged (%s), reset!",
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

def get_pickle_folder(path):
    """check pickle folder existence"""
    pickle_folder = os.path.join(path, "imgdups")
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    return pickle_folder

def scale_image(image):
    """scale image temporary for comparison"""
    if np.shape(image) != (500, 500, 1):
        image = cv2.resize(image, (500, 500))

    return image

def get_pickle(path):
    """Create a pickle file from target path"""
    pickle_file = os.path.join(get_pickle_folder(path), "image_cache.pkl")
    processed_files, index = clean_up_pickle(path, load_pickle(pickle_file))

    index_check = False

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        image_path = os.path.join(path, file)
        if not file in processed_files:
            image_target = cv2.imread(image_path)
            if image_target is None:
                continue

            image_target = scale_image(image_target)

            orb = cv2.ORB_create()
            _keypoints, descriptors = orb.detectAndCompute(image_target, None)
            index.append((image_path, descriptors))
            processed_files.append(file)
            logging.debug("Add processed file %s", file)
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

        search_image = scale_image(search_image)

        orb = cv2.ORB_create()
        _keypoints, search_descriptors = orb.detectAndCompute(search_image, None)

        duplicate_found = False
        match_score = 0
        match_highest_score = 0

        for target_filepath, target_descriptors in target_index:
            if os.path.basename(target_filepath) == "imgdups":
                continue
            bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            match_score = bf_match.match(search_descriptors, target_descriptors)
            if len(match_score) > 350:
                logging.info("%s == %s (score: %d)",
                        filename,
                        os.path.basename(target_filepath),
                        len(match_score))
                duplicate_found = True
                exit_status += 1
                break
            if len(match_score) > match_highest_score:
                match_highest_score = len(match_score)

        # If no duplicate has been found after checking all target images, log the message
        if not duplicate_found:
            logging.info("No duplicate found for %s (score: %d)", filename, match_highest_score)

    logging.info("Script finished!")
    if exit_status:
        sys.exit(exit_status)

if __name__ == "__main__":
    main()
