#!/usr/bin/env python

""" Script to search reference images for duplicates in a target path """

# common
import os
import sys
import logging
import pickle

from argparse import ArgumentParser

# opencv
import cv2
import numpy as np

# configure logger
logger = logging.getLogger('imgdups')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_pickle_folder(path):
    """check pickle folder existence"""
    pickle_folder = os.path.join(path, "imgdups")
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    return pickle_folder

def validate_file_extension(file):
    """Check valid image file extensions"""
    validate = False
    if file.lower().endswith(('.jpg', '.png', 'jpeg', '.bmp')):
        validate = True

    return validate

def get_files_from_path(path):
    """walk through directory and return files list"""
    files = []
    for file in os.listdir(path):
        if ( os.path.isfile(os.path.join(path, file))
                and validate_file_extension(file) ):
            files.append(file)

    return files

def scale_image(image):
    """scale image temporary for comparison"""
    if np.shape(image) != (500, 500, 1):
        image = cv2.resize(image, (500, 500))

    return image

def get_descriptors(path):
    """get descriptors from image for comparison"""
    image = cv2.imread(path)
    if image is None:
        return False
    image = scale_image(image)
    orb = cv2.ORB_create()
    _keypoints, descriptors = orb.detectAndCompute(image, None)

    return descriptors

def load_cache_index(file):
    """Load cache from pickle file"""
    index = []
    processed_files = []

    if os.path.exists(file):
        logger.debug("Cache file %s found, load existent object structure", file)
        with open(file, 'rb') as feat:
            try:
                processed_files, index = pickle.load(feat)
                logger.debug("Processed files in cache found: %d", len(processed_files))
            except (EOFError, ValueError) as ex:
                logger.debug("Cache file %s found but damaged (%s), reset!",
                        file, str(ex))
                os.remove(file)

    return processed_files, index

def rebuild_cache_index(path, index):
    """check file path exists and clean up pickle data"""
    clean_index = []
    clean_processed_files = []
    for file, data in index:
        if "imgdups" in file or "thumb" in file:
            continue

        original_path = os.path.join(path, os.path.basename(file))
        if ( file not in clean_index
                and os.path.exists(original_path) ):
            # copy old index into new index after file check
            clean_index.append((file, data))
            clean_processed_files.append(os.path.basename(file))

    return clean_processed_files, clean_index

def check_garbage(file_path):
    """check for empty files"""
    check = False
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        check = True

    return check

def main():
    """Start standalone with config file or arguments"""
    parser = ArgumentParser()
    parser.add_argument("-s", "--search", dest="search_path", required=True,
                        help="path to images folder for duplicate candidates", metavar="PATH")
    parser.add_argument("-t", "--target", dest="target_path", required=True,
                        help="path to images folder to check for duplicates", metavar="PATH")
    parser.add_argument("-m", "--match", dest="match_score",
                        help="min match result score to return image as duplicate", metavar="INT")

    # parse args and ignore unknown args
    args, _unknown = parser.parse_known_args()

    # use argument -m or set default score
    match = 320
    if args.match_score.isdigit():
        # test for maximum possible match score
        if int(args.match_score) <= 500:
            match = int(args.match_score)

    img_dups = ImgDups(args.target_path, args.search_path)
    img_dups.find_duplicates(match)

class ImgDups():
    """
    Class to find image duplicates
    in target path from a search path
    """
    def __init__(self, target, search):
        self.target = target
        self.search = search
        self.duplicates = []
        self.image_cache = []
        self.image_processed = []
        self.search_cache = []
        self.search_processed = []

    def get_stats(self):
        """Return stats"""
        image_cache_path = self.get_image_cache_path()

        # size in MB
        if os.name == 'nt':
            image_cache_size = os.path.getsize(image_cache_path) / 1024 ** 2
        else:
            image_cache_size = os.path.getsize(image_cache_path) / 1000 ** 2

        stats = {
                "image_processed": len(self.image_processed),
                "image_cache_size_mb": round(image_cache_size, 3),
                "search_processed": len(self.search_processed),
                "duplicates": len(self.duplicates)
        }

        return stats

    def get_image_cache_path(self):
        """Return image cache file path"""
        return os.path.join(get_pickle_folder(self.target), "image_cache.pkl")

    def get_search_cache_path(self):
        """Return search cache file path"""
        return os.path.join(get_pickle_folder(self.search), "dup_cache.pkl")

    def get_image_cache(self, path):
        """
        Load existent pickle data from target path,
        dump updated pickle data to file,
        return file path
        """
        pickle_file = self.get_image_cache_path()
        self.image_processed, self.image_cache = load_cache_index(pickle_file)
        self.image_processed, self.image_cache = rebuild_cache_index(path, self.image_cache)

        index_check = False

        files = get_files_from_path(path)
        for file in files:
            # skip folder imgdups and thumb files
            if ( "imgdups" in file
                    or "thumb" in file ):
                continue

            image_path = os.path.join(path, file)
            if not file in self.image_processed:
                descriptors = get_descriptors(image_path)
                self.image_cache.append((image_path, descriptors))
                self.image_processed.append(file)
                logger.debug("Add processed file %s", file)
                index_check = True

        if index_check:
            # Save processed files and their features
            with open(pickle_file, 'wb') as feat:
                logger.info("Write new cache file (%d images)", len(self.image_processed))
                pickle.dump((self.image_processed, self.image_cache), feat)

        return pickle_file

    def cleanup_search_cache(self, filename):
        """
        Remove obsolete images from search cache
        and check if it is already known, return check state
        """
        check = False
        search_filepath = os.path.join(self.search, filename)
        search_filesize = os.path.getsize(search_filepath)

        # clean up processed files cache
        for search_proc_file in self.search_processed:
            if not os.path.exists(os.path.join(self.search, search_proc_file)):
                self.search_processed.remove(search_proc_file)

        # clean up index data
        for search_file in self.search_cache:
            if ( not os.path.exists(search_file[0])
                    or (search_filepath == search_file[0]
                        and search_filesize != search_file[1]) ):
                # clean up cache
                self.search_cache.remove(search_file)

            elif ( search_filepath == search_file[0]
                    and search_filesize == search_file[1] ):
                # cache is fine
                check = True

        if not check or not filename in self.search_processed:
            logger.debug("Add processed file to search cache %s", filename)
            self.search_processed.append(filename)
            check = False
        else:
            logger.info("Skip duplicate check for %s", filename)

        return check

    def save_search_cache(self):
        """Remove old cache and write updated cache to file"""
        file_path = self.get_search_cache_path()
        if os.path.exists(file_path):
            # clean up
            os.remove(file_path)

        with open(file_path, 'wb') as cache_file:
            # write duplicates checked to cache file for next run
            pickle.dump((self.search_processed, self.search_cache), cache_file)

    def find_duplicates(self, match = 320):
        """ Start the script """
        logger.info("Start script")

        garbage = 0

        if not os.path.exists(self.target):
            logger.error("Target path does not exist (%s)", self.target)
            sys.exit(1)

        self.get_image_cache(self.target)
        self.search_processed, self.search_cache = load_cache_index(self.get_search_cache_path())

        logger.info("Search path: %s", self.search)
        logger.info("Target path: %s", self.target)
        logger.info("Starting image comparison, search <-> target")

        search_files = get_files_from_path(self.search)
        for filename in search_files:
            duplicate_found = False
            file_path = os.path.join(self.search, filename)

            if self.cleanup_search_cache(filename):
                continue

            search_filesize = os.path.getsize(file_path)
            self.search_cache.append((file_path, search_filesize))

            search_descriptors = get_descriptors(file_path)

            match_score = 0
            match_score_high = 0
            match_score_name = ""

            for target_filepath, target_descriptors in self.image_cache:
                target_filename = os.path.basename(target_filepath)

                if check_garbage(target_filepath):
                    garbage += 1
                    continue

                bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # calculate match score
                match_score = bf_match.match(search_descriptors, target_descriptors)

                if len(match_score) > match:
                    logger.info("%s == %s (score: %d)",
                            filename,
                            target_filename,
                            len(match_score))

                    self.duplicates.append({
                        "search": filename,
                        "target": target_filename,
                        "score": len(match_score),
                        "size": search_filesize
                    })

                    duplicate_found = True
                    break

                if len(match_score) > match_score_high:
                    match_score_high = len(match_score)
                    match_score_name = target_filename

            if not duplicate_found:
                logger.info("%s != %s (score: %d)",
                        filename, match_score_name, match_score_high)

        self.save_search_cache()

        if garbage > 0:
            logger.warning("Ignored %d garbage files in target folder", garbage)

        logger.info("Script finished with %d duplicates found!", len(self.duplicates))

        return self.duplicates

if __name__ == "__main__":
    main()
