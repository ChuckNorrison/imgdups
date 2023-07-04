# imgdups
[![Pylint](https://github.com/ChuckNorrison/imgdups/actions/workflows/pylint.yml/badge.svg)](https://github.com/ChuckNorrison/imgdups/actions/workflows/pylint.yml)

Most image duplicate checkers can find duplicates within a single folder. This solution can verify that no duplicates from one path (search) exists in another path (target). It will use opencv to create image descriptors and cache them into a pickle file for faster processing after it was run the first time. With this approach we can not just find exact duplicates but similar images based on a match score.

## Requirements
Python 3.6+ was tested

`sudo apt install python3 python3-pip`

## Option 1: Install from Source
```
git clone https://github.com/ChuckNorrison/imgdups
cd imgdups
pip3 install .
```

## Option 2: Install from PyPi (recommended)
`pip3 install imgdups`

## CLI Usage
`imgdups --search "/path/to/reference" --target "/path/to/check"`

or if not installed (git clone first)

```
cd imgdups
python3 imgdups.py --search "/path/to/reference" --target "/path/to/check"
```

## Python example

```
#!/usr/bin/env python3
import imgdups

SEARCH_PATH = "/path/to/reference"
TARGET_PATH = "/path/to/check"

img_dups = imgdups.ImgDups(TARGET_PATH, SEARCH_PATH)
duplicates = img_dups.find_duplicates()

for duplicate in duplicates:
    print("%s == %s (score: %d)",
            duplicate["target"],
            duplicate["search"],
            duplicate["score"]
    )

print("%d duplicates found", len(duplicates))

```