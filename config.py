#!/usr/bin/env python

from datetime import datetime

# path to search duplicates in
TARGET_PATH = "target/photos"

# path for potential duplicates to search in target path
SEARCH_PATH = "search/photos"

# date to check for duplicates
CHECK_DATE = datetime.now().strftime("%d-%m-%Y")
#CHECK_DATE = "14-06-2023"
