""" 
Image Annotator Driver

Michael Massone
7180 Advanced Perception
Created: 2024/11/10
Updated: 2025/02/05

This is the driver file for the AnnotationManager 
"""
# Packages
import os
import argparse

# Modules
from old_version.image_processor_class import LogChromaticity
from annotator_class import AnnotationManager

# Argparser
parser = argparse.ArgumentParser(description="Select Folder for annotation. Example: 'folder_5'")
parser.add_argument('folder_name', type=str, help='The name of the folder')
parser.add_argument(
        '--save_map', 
        action='store_true',
        default=False,
        help='Skip saving ISD pixel map as TIFF file. By default, it will be saved.'
    )
    
args = parser.parse_args()

# This is name of the folder that contains the images to be processed
folder = args.folder_name
save_isd_map = args.save_map

############### NONE OF THIS NEEDS TO BE UPDATED!!!! #############################
image_dir = f'images/{folder}/'
isd_map_dir = f'annotations/{folder}/isd_maps/'
xml_file_path = f'annotations/{folder}/annotations.xml'


# Check if the directory exists
if not os.path.isdir(image_dir):
    print(f"Error: The directory {image_dir} does not exist.")
    exit(1)

if not os.path.exists(isd_map_dir):
    os.makedirs(isd_map_dir)
    print(f"Directory created at {isd_map_dir}")
else:
    print(f"Directory exists at {isd_map_dir}")

def main():
    try:
        image_annotator = AnnotationManager(image_dir, isd_map_dir, save_isd_map)
        image_annotator.set_xml_file(xml_file_path)
        image_annotator.annotate_images()
    except Exception as e:
        print(f"An error occurred during annotation: {e}")
        exit(1)

if __name__ == "__main__":
    main()