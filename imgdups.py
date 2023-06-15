#!/usr/bin/env python

""" Script to find duplicated images in a target path """

# common
import os
import sys
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

def load_config():
    """Load the config.py as module"""
    config = importlib.import_module('config')
    if ( not os.path.exists(config.TARGET_PATH)
            or not os.path.exists(config.SEARCH_PATH) ):
        logging.error("Config for TARGET_PATH or SEARCH_PATH is missing, Exit!")
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
    processed_files, target_index = [], []

    if os.path.exists(file):
        logging.info("Pickle file %s found, load existent object structure", file)
        with open(file, 'rb') as feat:
            try:
                processed_files, target_index = pickle.load(feat)
            except EOFError as ex:
                logging.debug("Pickle file %s found but damaged (%s), reset!",
                        file, str(ex))
                os.remove(file)

    return processed_files, target_index

def get_pickle_index(path):
    """Create a pickle file from target path"""
    pickle_file = "features.pkl"
    processed_files, index = load_pickle(pickle_file)
    index_check = False

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if not filename in processed_files:
            image_target = cv2.imread(file_path)

            if image_target.shape != (500, 500, 3):
                image_target = cv2.resize(image_target, (500, 500))
                # save scaled image
                if cv2.imwrite(file_path+".tmp", image_target):
                    # replace original
                    logging.debug("Replace old image: %s", filename)
                    os.remove(file_path)
                    os.rename(file_path+".tmp", file_path)

            orb = cv2.ORB_create()
            _kp, des = orb.detectAndCompute(image_target, None)
            index.append((file_path, des))
            processed_files.append(filename)
            index_check = True

    if index_check:
        # Save processed files and their features
        with open(pickle_file, 'wb') as feat:
            pickle.dump((processed_files, index), feat)

    return index

def main():
    """ Start the script """
    logging.info("Start with cv2 version: %s", cv2.__version__)

    config = load_config()

    # remove thumbs
    remove_thumbs(config.TARGET_PATH)
    remove_thumbs(config.SEARCH_PATH)

    target_index = get_pickle_index(config.TARGET_PATH)

    # Start comparison
    logging.info("Starting image comparison...")
    for _idx, filename in enumerate(os.listdir(config.SEARCH_PATH), 1):
        file_path = os.path.join(config.SEARCH_PATH, filename)
        search_image = cv2.imread(file_path)

        if search_image.shape != (500, 500, 3):
            search_image = cv2.resize(search_image, (500, 500))

        orb = cv2.ORB_create()
        _kp, search_des = orb.detectAndCompute(search_image, None)

        duplicate_found = False

        for target_filepath, target_des in target_index:
            bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_match.match(search_des, target_des)
            if len(matches) > 300:
                logging.info("%s == %s (score: %d)",
                        filename,
                        os.path.basename(target_filepath),
                        len(matches))
                duplicate_found = True
                break

        # If no duplicate has been found after checking all target images, log the message
        if not duplicate_found:
            logging.info("No duplicate found for %s", filename)

    logging.info("Script finished!")

if __name__ == "__main__":
    main()
