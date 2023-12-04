### VisDrone Dataset Aqquisition
This folder contains instructions to aqquire the VISDRONE dataset. The full dataset is too large to be stored in the github.
The folder structure is inlcuded in the repo, but the images are not

## Automated Script
You can run `aqquire_dataset.py` to have the python script do the downloading for you.
- Downloading the images directly from google drive within python has it's limitations. There are sometimes imposed limits that restrict this process.
- If you have errors with downloading them, you can manually download these 3 zip files to your computer, and place them inside `VisDrone_Dataset_Aqquisition/zip_files`. Then run the script normally.
    - `https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing`
    - `https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing`
    - `https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V`

## Note on the dataset description yaml file (VisDrone.yaml)
You will likely need to change the absolute path references to fit your directory structure.