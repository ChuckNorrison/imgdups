# imgdups
[![Pylint](https://github.com/ChuckNorrison/imgdups/actions/workflows/pylint.yml/badge.svg)](https://github.com/ChuckNorrison/imgdups/actions/workflows/pylint.yml)

Find image duplicates from a search path in a target path

## Setup
- `sudo apt install python3 python3-pip`
- `pip3 install -r requirements.txt`

## Usage
Edit `config.py` with your desired image folder paths. Start the script with `python3 imgdups.py`. 

## Advanced
Import the script in your own code and use it without a config file

```
from imgdups import ImgDups
img_dups = ImgDups("target_path", "search_path")
duplicates = img_dups.find_duplicates()
```
