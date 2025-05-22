""" 
Image Annotator Class

Nelson Farrell and Michael Massone
Created: 2024/11/10
Updated:2025/02/05


This file contains a class that allows a user to make annotations to an image and convert the image to log chromaticity space.

This code based on a previous annotator created by: Chang Liu & Yunyi Chi
"""
# Packages
import os
import cv2
import csv
import shutil
import tifffile
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom


# Modules
from old_version.image_processor_class import LogChromaticity

# Class to manage the annotation state for each image
class AnnotationManager:
    """
    Allows users to make annotations and processes the images in real time; i.e., log chromaticity.
    Annotations can be added with a left click and removed with a right click. 
    Other functionalities and saving is performed with key strokes.

    Annotate the lit area followed by the dark area! 
    """
    def __init__(self, image_dir, isd_maps_dir, save_isd_map=True):
        """
        Initializer for AnnotationManager

        Parameters: 
         * None
        """
        # Image click data
        self.click_count = 0
        self.clicks = []  # List to store (lit_row, lit_col, shad_row, shad_col) pairs

        # Directories
        self.image_folder = image_dir
        self.save_isd_map = save_isd_map
        self.isd_maps_dir = isd_maps_dir
        
        # XML file
        self.xml_file = None

        # Image of interest
        self.annotated_image_list = None # List of already annotated image filenames
        self.img = None
        self.image_path = None # Used to read a UNCHANAGED version for processing
        self.processed_img = None

        # Processing class object
        self.img_processor = None

    def set_image_processor(self):
        """
        Instantiates new image processor object.
        The image processor converts the image to log chromaticity space using the annotations
        """
        self.img_processor = LogChromaticity()

    def set_xml_file(self, xml_file_path) -> None:
        """
        Create the XML file with a root element if it does not exist; assigns to xml_file attribute.
        
        Parameters:
            xml_file_path (str): Path to the XML file.
        """
        # Check if the XML file already exists
        if not os.path.exists(xml_file_path):
            # Create the root element
            root = ET.Element("annotations")
            tree = ET.ElementTree(root)
            
            # Write the root element to the file to initialize it
            with open(xml_file_path, 'wb') as xmlfile:
                tree.write(xmlfile, encoding="utf-8", xml_declaration=True)
        
        # Assign the file path to the attribute
        self.xml_file = xml_file_path

    def is_complete(self) -> bool:
        """
        Check if a minimum of 1 pairs are annotated (12 clicks minimum).
        """
        if (self.click_count >= 2) and (self.click_count % 2 == 0):
            return True
        else:
            print("You must select at least one pair of lit/shadow pixels.")

    def show_message(self) -> None:
        """
        Show a message indicating which point is expected next.
        """
        if self.click_count > 0:
            pair_num = (self.click_count // 2) + 1
            if self.click_count % 2 == 0:
                print(f'Click lit patch for pair {pair_num}')
            else:
                print(f'Click shadow patch for pair {pair_num}')

    @staticmethod
    def prettify_element(element):
        """Return a prettified XML string for a single element."""
        raw_str = ET.tostring(element, encoding="unicode")
        parsed = xml.dom.minidom.parseString(raw_str)
        return parsed.toprettyxml(indent="  ").strip()

    def write_to_xml(self, image_name, status):
        """
        Writes image name, target directory, and annotations to a valid XML file.
        Ensures a single root element for the XML file.
        """
        if not image_name or not status:
            print("Error: image_name or target_directory is empty.")
            return

        try:
            # File exists: Load existing XML, else create a new root
            if os.path.exists(self.xml_file):
                tree = ET.parse(self.xml_file)
                root = tree.getroot()
                image_count = len(root.findall('.//image'))
                print(f"Number of processed images: {image_count}")
            else:
                root = ET.Element("annotations")
                tree = ET.ElementTree(root)

            # Add new image element
            image_element = ET.Element("image")
            image_element.set("name", image_name.lower())
            image_element.set("status", status)
            image_element.set("patch_size", self.img_processor.get_patch_size())
            image_element.set("anchor_point", self.img_processor.get_anchor_point())

            if self.clicks:
                for i, (lit_row, lit_col, shad_row, shad_col) in enumerate(self.clicks, start=1):
                    click_element = ET.SubElement(image_element, "click", id=str(i))
                    lit_element = ET.SubElement(click_element, "lit")
                    lit_element.set("row", str(lit_row))
                    lit_element.set("col", str(lit_col))
                    shad_element = ET.SubElement(click_element, "shadow")
                    if shad_row is not None:
                        shad_element.set("row", str(shad_row))
                    if shad_col is not None:
                        shad_element.set("col", str(shad_col))

            # Prettify the newly created image element
            pretty_image_element = self.prettify_element(image_element)

            # Append the prettified element to the root
            root.append(ET.fromstring(pretty_image_element))

            # Save the updated XML file
            tree.write(self.xml_file, encoding="unicode", xml_declaration=True)
            print(f"Data successfully written to {self.xml_file}")

        except Exception as e:
            print(f"Error while writing to XML: {e}")


    def process_image(self) -> None:
        """
        Processes the image using LogChromaticity.process_img()
        """
        # self.img_processor = LogChromaticity()
        self.processed_img = self.img_processor.process_img(self.image_path, self.clicks)

    def update_patch(self, size) -> None:
        """
        Updates the patch size around the annotations
        """
        if self.processed_img is not None:
            self.processed_img = self.img_processor.update_patch_size((size, size))
            self.display_images()

    def update_anchor(self, val) -> None:
        """
        Updates the anchor the point of the plane. 
        Controls brightness.
        """
        if self.processed_img is not None:
            self.processed_img = self.img_processor.update_anchor_point(val)
            self.display_images()
       
    
    def display_images(self, clickable=False) -> None:
        """
        Generates the display of the original image and the processed image. 
        """
        try:
            if self.processed_img is not None:
                combined_image = np.hstack((self.img, self.processed_img))
            else:
                combined_image = np.hstack((self.img, np.zeros_like(self.img)))

            window_name = "Original Image ---------------------------------------------------------------- Processed Image"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined_image)
            cv2.moveWindow(window_name, 0, 100)

            # IF needed for optimal window size
            # cv2.resizeWindow(window_name, 1000, 1000)
            
            if clickable:
                cv2.setMouseCallback(window_name, self.click_event)

        except Exception as e:
                    print(f"Error with self.display_images(): {e}")
        
    def reset(self) -> None:
        """
        Reset the annotation state for the current image.
        """
        self.click_count = 0
        self.clicks = []


    def add_click(self, row, col) -> None:
        """
        Add a click (light or shadow) based on the click count.

        Parameters:
            row (int): Click location row (y) index value.
            col (int): Click location col (x) index value.
        """
        if self.click_count % 2 == 0:
            # Add a new lit point
            self.clicks.append((row, col, None, None))  # Placeholder for shadow
        else:
            # Add shadow point to the last lit point
            self.clicks[-1] = (self.clicks[-1][0], self.clicks[-1][1], row, col)
        self.click_count += 1

    def remove_click(self) -> None:
        """
        Removes the last recorded click.
        - If the last click was a shadow point, it removes just the shadow point.
        - If the last click was a lit point without a shadow point, it removes the entire lit point entry.
        """
        if self.click_count == 0:
            print("No clicks to remove.")
            return
        
        # remove current annotation pair
        self.clicks.pop()

        # update click count
        if self.click_count % 2 == 1:
            self.click_count -= 1
        if self.click_count % 2 == 0:
            self.click_count -= 2      


    # Mouse event callback function
    def click_event(self, event, x, y, flags, params):
        """
        Records left or right click event. Left click selects a pixel for annotation, right click removes the most recent pixel or pixel pair. 

        Parameters:
            x (int): CLick location x coordinate.
            y (int): click lcoation y coordinate.

        """
        if event == cv2.EVENT_LBUTTONDOWN:
            row, col = y, x  # Convert (x, y) to (row, col)
            self.add_click(row, col)

            # Alternate between two colors for the circles ~ Green for light, Red for shadow
            color = (0, 0, 255) if self.click_count % 2 == 0 else (0, 255, 0)
            cv2.circle(self.img, (x, y), 5, color, 2)

            # Draw a line between every pair of clicks
            if self.click_count % 2 == 0:
                cv2.line(self.img, (self.clicks[-1][1], self.clicks[-1][0]),
                         (self.clicks[-1][3], self.clicks[-1][2]), (255, 255, 255), 2)

            # cv2.imshow('image', self.img)
            self.display_images()

            # Process image after every pair of click have been made
            if self.click_count >= 2 and self.click_count % 2 == 0:
                self.process_image()
                self.display_images()
            self.show_message()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_click()

            # Redraw the image to reflect the removed click
            self.img = cv2.imread(self.image_path)  # Reload original image to clear previous annotations
            for idx, (lit_row, lit_col, shad_row, shad_col) in enumerate(self.clicks):
                lit_color = (0, 255, 0)  # Green for lit points
                shad_color = (0, 0, 255)  # Red for shadow points
                cv2.circle(self.img, (lit_col, lit_row), 5, lit_color, 2)

                if shad_row is not None and shad_col is not None:
                    cv2.circle(self.img, (shad_col, shad_row), 5, shad_color, 2)
                    cv2.line(self.img, (lit_col, lit_row), (shad_col, shad_row), (255, 255, 255), 2)      
            
            # Update display images
            if self.click_count >= 2 and self.click_count % 2 == 0:
                self.process_image()            
            self.display_images()

    def save_pixel_map(self, image_name) -> None:
        """
        Save the pixel map to .tiff.

        Note: The isd values are multiplied by 65535 before saving. When using the isd_maps the user has 
        divide the isd_map by 65535 and map to a float. 
        """
        pixel_map = self.img_processor.get_isd_pixel_map()
        print(f"Pixel map data type: {pixel_map.dtype}")
        pixel_map = pixel_map * 65535
        image_name_no_ext = os.path.splitext(image_name)[0]
        save_path = os.path.join(self.isd_maps_dir, f"{image_name_no_ext}_isd.tiff")
        # pixel_map_image = Image.fromarray(pixel_map)
        # image.save(save_path, format="TIFF")
        tifffile.imwrite(save_path, pixel_map)
        print(f"ISD Pixel Map saved at {save_path}")


    def export_completed_image(self, image_name, status) -> None:
        """
        Moves the current image to its destination folder, saves the isd_map, and updates the XML doc.
        """
        if status == "completed":
            print(f"Saving processed ISD map {self.isd_maps_dir}.")
            self.write_to_xml(image_name, status)
            if self.save_isd_map:
                self.save_pixel_map(image_name)
            self.processed_img = None

        if status == "dropped":
            print(f"Dropping{image_name}.")
            self.write_to_xml(image_name, status)
            self.processed_img = None


    def image_previously_annotated(self, image_name):
        """
        Checks to see if the current image has already been annotated.
        """
        if os.path.exists(self.xml_file):
            if self.annotated_image_list is None:
                try:
                    tree = ET.parse(self.xml_file)
                    root = tree.getroot()
                    self.annotated_image_list = [image.get("name") for image in root.findall(".//image")]
                except (FileNotFoundError, ET.ParseError) as e:
                    print(f"Error loading XML file: {e}")
                    return False

            if image_name in self.annotated_image_list:
                print(f"Image: {image_name} already annotated.")
                return True
        return False

    def annotate_images(self):
        """
        Loops over images in a dir allowing user to annotate images.
        """
        for image_name in os.listdir(self.image_folder):
            if image_name.endswith(('tif', 'tiff')):
                
                # Check if image filename exists in annotation XML already
                if self.image_previously_annotated(image_name):
                    continue
                
                # Set image path
                self.image_path = os.path.join(self.image_folder, image_name)

                try:
                    # Init image processing class
                    self.set_image_processor()

                    # Read image
                    self.img = cv2.imread(self.image_path)
                    if self.img is None:
                        print(f"Cannot open image: {image_name}.")
                        continue

                    # Sets click count to zero
                    self.reset() 

                    # GUI
                    self.display_images(clickable=True)
                    cv2.namedWindow("Log Space Widget")
                    cv2.createTrackbar("Anchor Point", "Log Space Widget", 104, 111, self.update_anchor)
                    cv2.createTrackbar("Patch Size", "Log Space Widget", 1, 61, self.update_patch)

                    # Instructions
                    print("""Press: 
                            \n    'Enter' to save image annotations 
                            \n    'SPACE' to drop the image. 
                            \n    'r' to redo the annotations
                            \n    'q' to quit the annotator.
                            """)  

                    # Annotation loop
                    while True:

                        key = cv2.waitKey(0)

                        if key == 13 and self.is_complete():
                            print(f"Saving annotations for {image_name}.")
                            self.export_completed_image(image_name, "completed")
                            break  # Move to the next image
                        
                        elif key == 32:
                            print(f"Dropping {image_name}.")
                            self.export_completed_image(image_name, "dropped")
                            break

                        elif key == ord('r'):
                            print(f"Starting over for {image_name}. Redo the annotations.")
                            self.img = cv2.imread(self.image_path)  # Reload the image to clear drawn points
                            self.reset() # Sets click count to zero
                            self.display_images(clickable=True)

                        elif key == ord('q'):
                            print("Quitting.")
                            cv2.destroyAllWindows()
                            return

                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"An error occurred with image {image_name}: {e}")
        print("All images processed and data saved.")

##################################################################################################################################
if __name__ == "__main__":
    pass
