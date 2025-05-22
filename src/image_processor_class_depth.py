""" 
Log Space Chromaticity Image Processor

Michael Massone
Created: 2024/11/10
Updated: 2025/05/22

This is class take in a 16bit image and create a log space chromaticity image and an ISD pixel map and Surface Normnal segmentation. 
"""

import numpy as np
import cv2 
import logging

class DepthBasedLogChromaticity:
    """
    Converts an image to log chromaticity space using user annotations
    """
    def __init__(self) -> None:
        """
        Initializes LogChromaticity
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.method = 'mean'
        self.anchor_point= np.array([10.8, 10.8, 10.8])
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        # Default patch size set to single pixel value
        self.patch_size = (11, 11)

        self.lit_pixels = None
        self.shadow_pixels = None
        self.mean_isd = None

        # Map matrices
        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None

    def reset_state(self) -> None:
        """
        Resets the internal state of the LogChromaticity instance.
        This is useful when reusing the same instance for a new image.
        """
        self.logger.info("Resetting LogChromaticity state.")

        self.method = 'mean'
        self.anchor_point = np.array([10.8, 10.8, 10.8])
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        self.patch_size = (11, 11)

        self.lit_pixels = None
        self.shadow_pixels = None
        self.mean_isd = None

        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None

    def set_img_rbg(self, image_path) -> None:
        """
        Set the image of interest

        Parameters:
            * image_path: (str) - Path the image of interest. 
        """
        self.rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        return None
    
    def set_lit_shadow_pixels_list(self, clicks) -> None:
        """
        Sets the lit and shadow pixels.

        Parameters:
            * clicks: (list) - Annotations coordinates from the annotator 
        """
        self.logger.debug(f"Clicks: {clicks}")
        self.lit_pixels = [(pair[0], pair[1]) for pair in clicks]
        self.logger.debug(f"Lit Pixels: \n{self.lit_pixels}")
        self.shadow_pixels = [(pair[2], pair[3]) for pair in clicks]
        self.logger.debug(f"Shadow Pixels: \n{self.shadow_pixels}")

        return None
    
    def get_patch_size(self):
        """
        Returns patch size.
        """
        return self.patch_size
    
    def get_isd_pixel_map(self):
        """
        Returns ISD pixel map as numpy array.
        """
        return self.isd_map
    
    def get_anchor_point(self):
        """
        Returns anchor point as np.array.
        """
        return self.anchor_point

    ###################################################################################################################
    # Methods for Pre Processing Images
    ###################################################################################################################

    def convert_img_to_log_space(self) -> None:
        """
        Converts a 16-bit linear image to log space, setting linear 0 values to 0 in log space.

        Parameters:
        -----------
        img : np.array
            Input 16-bit image as a NumPy array.

        Returns:
        --------
        log_img : np.array
            Log-transformed image with 0 values preserved.
        """

        log_img = np.zeros_like(self.rgb_img, dtype = np.float32)
        log_img[self.rgb_img != 0] = np.log(self.rgb_img[self.rgb_img != 0])

        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        self.log_img = log_img

        return None
    
    
    ###################################################################################################################
    # Methods for using Weighted Mean
    ###################################################################################################################
    def get_patch_mean(self, center, patch_size):
        """
        Extracts a patch from a NumPy array centered at a specific pixel location 
        and returns the mean of the patch.

        Parameters:
        - img: np.array, the input array (can be 2D or 3D).
        - center: tuple, the (y, x) coordinates of the center pixel.
        - patch_size: tuple, the (height, width) of the patch.

        Returns:
        - mean_value: float, the mean of the extracted patch.
        """
        y, x = center
        patch_height, patch_width = patch_size

        # Calculate the start and end indices, ensuring they don't go out of bounds
        start_y = max(y - patch_height // 2, 0)
        end_y = min(y + patch_height // 2 + 1, self.log_img.shape[0])
        start_x = max(x - patch_width // 2, 0)
        end_x = min(x + patch_width // 2 + 1, self.log_img.shape[1])

        # Extract the patch and return its mean value
        patch = self.log_img[start_y:end_y, start_x:end_x]
        # self.logger.debug(f"Patch: {patch}")
        # self.logger.debug(f"Patch shape: {patch.shape}")
        mean_value = np.mean(patch, axis=(0, 1))
        # self.logger.debug(f"Patch mean: {mean_value}")

        return mean_value


    def get_annotation_map(self) -> None:
        """
        Creates an ISD map matching image dimensions.
        At each midpoint between a lit-shadow pair, stores the ISD vector (mean difference of RGB patches).
        All other pixels are set to NaN.
        """
        # Ensure valid pairs
        valid_pairs = [
            ((x1, y1), (x2, y2))
            for (x1, y1), (x2, y2) in zip(self.lit_pixels, self.shadow_pixels)
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None
        ]

        if not valid_pairs:
            raise ValueError("No valid lit-shadow annotation pairs found.")

        # Compute ISDs and midpoints
        isd_values = []
        midpoints = []
        for (x1, y1), (x2, y2) in valid_pairs:
            self.logger.debug(f"Pair: {(x1, y1), (x2, y2)}")
            lit_mean = self.get_patch_mean((y1, x1), self.patch_size)
            shadow_mean = self.get_patch_mean((y2, x2), self.patch_size)
            spectral_ratios = lit_mean - shadow_mean
            self.logger.debug(f"Spectral ratios: {spectral_ratios}")
            norms = np.linalg.norm(spectral_ratios, axis=0)
            isd = spectral_ratios / norms
            self.logger.debug(f"ISD: {isd}")
            # midpoint = (int((x1 + x2) // 2), int((y1 + y2) // 2))  # (x, y)
            isd_values.append(isd)
            midpoints.append((x1, y1))

        # Create map filled with NaNs
        height, width = self.rgb_img.shape[:2]
        isd_map = np.full((height, width, 3), np.nan, dtype=np.float32)

        # Assign ISD values to midpoints
        for (x, y), isd in zip(midpoints, isd_values):
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            isd_map[y, x, :] = isd

        self.annotation_map = isd_map
        self.logger.debug(f"\nAnnotation map values:  {np.unique(isd_map.reshape(-1, isd_map.shape[2]), axis=0)[:4]}")

    # def propagate_annotations(self, segmentation_map):
    #     """
    #     Propagate annotation values to all pixels in their segment.

    #     Args:
    #         annotation_map (np.ndarray): Same shape as segmentation_map. 
    #                                     Pixels with annotations have values, others are np.nan or a sentinel.
    #         segmentation_map (np.ndarray): 2D array with segment IDs per pixel.

    #     Returns:
    #         np.ndarray: New map with propagated annotation values.
    #     """
    #     self.segmentation_map = segmentation_map
    #     output = np.full_like(self.annotation_map, np.nan, dtype=np.float32)
    #     unique_segments = np.unique(segmentation_map)
    #     self.logger.debug(f"Unique segments: {unique_segments}")

    #     for seg_id in unique_segments:
    #         self.logger.debug(f"Segment: {seg_id}")
    #         mask = (segmentation_map == seg_id)
    #         values = self.annotation_map[mask, :]
    #         annotated_vals = values[~np.isnan(values).any(axis=1)]
    #         self.logger.debug(f"Annotated values: \n{annotated_vals}")

    #         if len(annotated_vals) > 0:
    #             if annotated_vals.ndim == 1:
    #                 mean_val = annotated_vals  # Already a single RGB triplet
    #             else:
    #                 mean_val = np.mean(annotated_vals, axis=0)
    #             self.logger.debug(f"Mean val: {mean_val}")
    #             output[mask] = mean_val

    #     # If any pixels are still NaN (segments with no annotation), fill with global mean
    #     global_annotated_vals = self.annotation_map[~np.isnan(self.annotation_map).any(axis=2)]
    #     global_mean = np.mean(global_annotated_vals, axis=0) if len(global_annotated_vals) > 0 else 0
    #     self.logger.debug(f"Global mean: {global_mean}")
    #     nan_mask = np.isnan(output).any(axis=-1)  # shape (H, W), bool
    #     output[nan_mask] = global_mean  # global_mean must be shape (3,)
    #     self.logger.debug(f"\nISD map values: \n{np.unique(output.reshape(-1, output.shape[2]), axis=0)}")
    #     self.isd_map = output

    def propagate_annotations(self, segmentation_map, eps=1e-6):
        """
        Propagate ISD annotation values to all pixels in each segment using
        inverse distance weighting via OpenCV's distance transform.

        If a segment contains:
            - 0 annotations: fill with global mean.
            - 1 annotation: assign that value directly to all pixels.
            - >1 annotations: weighted combination using inverse distance.

        Args:
            segmentation_map (np.ndarray): 2D array of shape (H, W) with segment IDs.

        Returns:
            None. Updates self.isd_map.
        """
        self.segmentation_map = segmentation_map
        output = np.full_like(self.annotation_map, np.nan, dtype=np.float32)
        unique_segments = np.unique(segmentation_map)

        for seg_id in unique_segments:
            mask = (segmentation_map == seg_id)
            coords = np.argwhere(mask)

            # Create mask of annotated pixels in this segment
            annotated_mask = np.zeros(mask.shape, dtype=np.uint8)
            annotated_values = []

            for (y, x) in coords:
                val = self.annotation_map[y, x]
                if not np.isnan(val).any():
                    annotated_mask[y, x] = 1
                    annotated_values.append(((y, x), val))

            if len(annotated_values) == 0:
                continue  # No annotations, fill later with global mean
            elif len(annotated_values) == 1:
                # Only one annotated point — assign it to all pixels
                output[mask] = annotated_values[0][1]
                continue

            # Compute distance transform for each annotated point and accumulate
            H, W = mask.shape
            segment_output = np.zeros((H, W, 3), dtype=np.float32)
            weight_map = np.zeros((H, W), dtype=np.float32)

            for (y, x), val in annotated_values:
                ann_mask = np.ones_like(mask, dtype=np.uint8)
                ann_mask[y, x] = 0  # Distance transform computes dist from 0s

                dist = cv2.distanceTransform(ann_mask, distanceType=cv2.DIST_L2, maskSize=5)
                dist = dist + eps  # Avoid division by zero
                weights = 1.0 / dist

                weights[~mask] = 0  # Ensure we only weight inside the segment

                for c in range(3):
                    segment_output[..., c] += weights * val[c]
                weight_map += weights

            # Normalize
            for c in range(3):
                segment_output[..., c] /= (weight_map + eps)

            output[mask] = segment_output[mask]

        # Fill any remaining NaNs with global mean
        global_annotated_vals = self.annotation_map[~np.isnan(self.annotation_map).any(axis=2)]
        global_mean = np.mean(global_annotated_vals, axis=0) if len(global_annotated_vals) > 0 else 0
        output[np.isnan(output).any(axis=-1)] = global_mean

        self.isd_map = output

    def project_to_plane_locally(self) -> None:
        """
        Projects each pixel to a plane orthogonal to the ISD for that pixel. 
        """
        shifted_log_rgb = self.log_img - self.anchor_point
        dot_product_map = np.einsum('ijk,ijk->ij', shifted_log_rgb, self.isd_map)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * self.isd_map

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        projected_rgb += self.anchor_point

        self.img_chroma = projected_rgb
    
    ###################################################################################################################
    # Methods for Post Processing Images
    ###################################################################################################################
    
    def log_to_linear(self) -> None:
        """
        Converts log transofrmed image back to linear space in 16 bits.

        Parameters:
        -----------
        log_rgb : np.array
            Log-transformed image with values between 0 and 11.1.

        Returns:
        --------
        vis_img : np.array
            Visualization-ready 8-bit image.
        """

        linear_img = np.exp(self.img_chroma)
        self.linear_converted_log_chroma = linear_img
        return None
    
    def convert_16bit_to_8bit(self) -> None:
        """
        Converts a 16-bit image to 8-bit by normalizing pixel values.

        Parameters:
        -----------
        img : np.array
            Input image array in 16-bit format (dtype: np.uint16).
        
        Returns:
        --------
        img_8bit : np.array
            Output image array converted to 8-bit (dtype: np.uint8).
        """
        #img_normalized = cv2.normalize(self.linear_converted_log_chroma, None, 0, 255, cv2.NORM_MINMAX) # divide by 255, clip to 0-255, then convert to unit8
        img_normalized = np.clip(self.linear_converted_log_chroma / 255, 0, 255)
        img_8bit = np.uint8(img_normalized)
        self.linear_converted_log_chroma_8bit = img_8bit
    
    ###################################################################################################################
    # Methods for using GUI Controls
    ###################################################################################################################

    def update_patch_size(self, size):
        """
        """
        self.patch_size = size
        self.logger.debug(f"Updated patch_size: {self.patch_size}")

        self.get_annotation_map()
        self.propagate_annotations(self.segmentation_map)
        self.project_to_plane_locally()
        self.log_to_linear()
        self.convert_16bit_to_8bit()
        return self.linear_converted_log_chroma_8bit

    def update_anchor_point(self, val):
        """
        """
        # Update the anchor point's first component based on trackbar position
        point = val / 10.0  # Scale value to range 0.0 to 11.1
        self.anchor_point = np.array([point, point, point])
        self.logger.debug(f"Updated anchor_point: {self.anchor_point}")

        self.project_to_plane_locally()
        self.log_to_linear()
        self.convert_16bit_to_8bit()

        return self.linear_converted_log_chroma_8bit
    
    ###################################################################################################################
    # Methods for  Execution
    ###################################################################################################################

    def process_img(self, image_input, clicks, segmentation_map) -> np.array:
        """
        Executes the full log-chromaticity processing pipeline:
        Converts a 16-bit RGB image to log space, computes ISD vectors from annotations, 
        propagates ISDs based on segmentation, projects to the ISD-orthogonal plane, 
        and converts the result back to an 8-bit linear RGB image.

        Args:
            image_input (Union[str, np.ndarray]):
                Either the file path to a 16-bit RGB image (str),
                or a pre-loaded image as a NumPy array of shape (H, W, 3), dtype=np.uint16.
            
            clicks (List[Tuple[int, int, int, int]]):
                A list of annotation tuples representing lit-shadow pixel pairs.
                Each tuple is of the form (lit_x, lit_y, shadow_x, shadow_y).
            
            segmentation_map (np.ndarray):
                A 2D array of shape (H, W), where each pixel contains an integer segment ID.
                Used to propagate ISD annotations spatially.

        Returns:
            np.ndarray:
                An 8-bit RGB image (dtype=np.uint8), representing the linearized version
                of the ISD-projected log-chromaticity image.
        """

        if isinstance(image_input, str):
            self.set_img_rbg(image_input)
        elif isinstance(image_input, np.ndarray):
            self.rgb_img = image_input
        else:
            raise TypeError("image_input must be a file path (str) or an image array (np.ndarray)")
        try:
            self.convert_img_to_log_space()
            self.set_lit_shadow_pixels_list(clicks)
            self.get_annotation_map()
            self.propagate_annotations(segmentation_map)
            self.project_to_plane_locally()
            self.log_to_linear()
            self.convert_16bit_to_8bit()
            return self.linear_converted_log_chroma_8bit
        except Exception as e:
            self.logger.warning(f"[ERROR] Failed during image processing: {e}")
            raise 

###################################################################################################################
    # TEST #
###################################################################################################################

def test_depth_chromaticity_pipeline():
    logging.basicConfig(
        format='%(name)s | %(levelname)s | %(message)s',
        level=logging.WARNING  # Default level for all loggers
    )
    logging.getLogger('DepthBasedLogChromaticity').setLevel(logging.DEBUG)
    # Import depth model
    encoder = 'vits'
    checkpoint_dir = 'src/depth_anything_v2/checkpoints'
    model_loader = DepthAnythingLoader(encoder=encoder, checkpoint_dir=checkpoint_dir)
    model = model_loader.get_model()

    # Run segmenter
    print("Testing")
    image_path = "/Users/mikey/Library/Mobile Documents/com~apple~CloudDocs/Code/code_projects/isd-annotator/images/example_folder/wenqing_fan_085.tif"
    segmenter = NormalSegmenter(model=model)
    segmentation_map, _ = segmenter.process_image(image_path, dynamic_k=True)

    # Example annotations: (x_lit, y_lit, x_shadow, y_shadow)
    # lit1, shadow1 = (70, 260), (70, 260,)
    # lit2, shadow2 = (1008, 685), (1008, 685)
    # lit3, shadow3 = (551, 197), (580, 193)
    # example_clicks = [(lit1, shadow1), (lit2, shadow2), (lit3, shadow3)]
    example_clicks = [
        (70, 260, 235, 135),
        (380, 740, 505, 590),
        (1008, 685, 969, 602),
        (551, 197, 580, 193),
        (1372, 114, 1403, 88),
        (594, 376, 638, 398),
        (1300, 650, 1200, 638)
    ]

    processor = DepthBasedLogChromaticity()
    chroma_image = processor.process_img(image_path, example_clicks, segmentation_map)

    # Assertions
    assert chroma_image is not None, "Processed image is None"
    assert chroma_image.shape == processor.rgb_img.shape, "Output shape mismatch"
    assert processor.log_img is not None, "Log image not computed"
    assert processor.isd_map is not None, "ISD map was not created"
    assert not np.all(np.isnan(processor.isd_map)), "ISD map contains only NaNs"

    
    # Visualization (optional)
    orig_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_normalized = np.clip(orig_image / 255, 0, 255)
    orig_img_8bit = np.uint8(img_normalized)
    
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(orig_img_8bit, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Log Chroma")
    plt.imshow(cv2.cvtColor(chroma_image, cv2.COLOR_BGR2RGB))
    for i, (x_lit, y_lit, x_shadow, y_shadow) in enumerate(example_clicks):
        plt.scatter(x_lit, y_lit, c='green', s=20, edgecolors='black', label='Lit' if i == 0 else "")
        plt.scatter(x_shadow, y_shadow, c='red', s=20, edgecolors='white', label='Shadow' if i == 0 else "")
        plt.plot([x_lit, x_shadow], [y_lit, y_shadow], c='white', linestyle='--', linewidth=1)
    plt.axis('off')

    # Assume isd_map is (H, W, 3), dtype=np.uint8
    isd_map = processor.isd_map.copy()
    h, w, _ = isd_map.shape
    flat = isd_map.reshape(-1, 3)

    # Get unique RGB triplets and assign each one an index
    unique_isds, inverse = np.unique(flat, axis=0, return_inverse=True)
    for i, isd in enumerate(unique_isds):
        print(f"INDEX: {i+1}  |  ISD: {isd}")
    indexed_img = inverse.reshape(h, w)

    # Generate distinct colors for display
    num_classes = unique_isds.shape[0]
    default_colors = plt.cm.tab10.colors  # Use a known colormap (tab10 has 10 distinct colors)
    display_colors = [default_colors[i % len(default_colors)] for i in range(num_classes)]
    cmap = ListedColormap(display_colors)

    handles = []
    for i, (isd, color) in enumerate(zip(unique_isds, display_colors)):
        label = f"ISD: [{isd[0]:.3f}, {isd[1]:.3f}, {isd[2]:.3f}]"
        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)

    # Show visualization
    plt.subplot(1, 3, 3)
    plt.imshow(indexed_img, cmap=cmap, interpolation='nearest')
    plt.title("ISD Segmentation")
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.05, -0.05), title="ISD Legend", fontsize='small')
    plt.axis('off')
    plt.colorbar(ticks=range(num_classes), label='RGB Index', shrink=0.25)
    plt.show()


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches


    from depth_class import DepthAnythingLoader
    from surface_normal_class import NormalSegmenter

    test_depth_chromaticity_pipeline()
###################################################################################################################   




    