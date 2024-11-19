""" 
Image Annotator Driver

Nelson Farrell and Michael Massone
7180 Advanced Perception
November 10, 2024
Bruce Maxwell, PhD.

This is the driver file for the AnnotationManager 
"""
# Modules
from image_processor_class import LogChromaticity
from annotator_class import AnnotationManager

# UPDATE THIS: This is name of the folder that contains the images to be processed
image_folder = 'new_annotator_dev/test_images'

############### NONE OF THIS NEEDS TO BE UPDATED!!!! #############################
# These are destination directories and files
# Directories
high_quality_dir = f'{image_folder}/processed/high_quality'
low_quality_dir = f'{image_folder}/processed/low_quality'
duplicates_dir = f'{image_folder}/processed/duplicates'
image_error_dir = f'{image_folder}/processed/image_error'
image_drop_dir = f'{image_folder}/processed/bad_images'
isd_maps_dir = f'{image_folder}/processed/isd_maps'

# Files
csv_file_path = f'{image_folder}/annotations_csv.csv'
xml_file_path = f'{image_folder}/annotations.xml'

directories = {
    "image_folder": image_folder,
    "high_quality_dir": high_quality_dir,
    "low_quality_dir":low_quality_dir,
    "duplicates_dir": duplicates_dir,
    "image_error_dir": image_error_dir,
    "image_drop_dir": image_drop_dir,
    "isd_maps_dir": isd_maps_dir
}

image_annotator = AnnotationManager()
image_annotator.set_directories(directories)
image_annotator.set_cvs_file(csv_file_path = csv_file_path)
image_annotator.set_xml_file(xml_file_path)
image_annotator.annotate_images()