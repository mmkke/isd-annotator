import cv2
import time
from pathlib import Path
import logging
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib             import colors as mcolors
from sklearn.decomposition  import PCA
from collections            import deque
from kneed                  import KneeLocator

# Modules
from depth_class            import DepthAnythingLoader

class NormalSegmenter:
    """
    A class to compute surface normals from a depth map, segment them into clusters
    based on normal direction, and visualize or contour the resulting segments.
    
    Attributes:
        model (callable): A depth estimation model with an `infer_image(image)` method.
        device (str): Target device (CPU/GPU).
        max_k (int): Number of clusters for KMeans segmentation.
    """

    def __init__(self, model=None, max_k=6, seed=18):
        """
        Initialize the NormalSegmenter.

        Args:
            model (optional): A depth estimation model that accepts an image and returns a depth map.
            k (int): Number of clusters for segmentation.
        """
        self.image_path = None
        self.image = None
        self.model = model
        self.max_k = max_k
        self.optimal_k = None
        self.rng = np.random.default_rng(seed)  # Seeded, or random if seed is None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.colormap = [
                        "#39FF14",  # Neon Green
                        "#FF073A",  # Neon Red
                        "#0FF0FC",  # Neon Cyan
                        # "#FF6EC7",  # Neon Pink
                         "#F5FE00",  # Neon Yellow
                        # "#7DF9FF",  # Electric Blue
                        # "#FF61AB",  # Hot Pink
                        # "#FE019A",  # Magenta
                        # "#16F529",  # Bright Lime
                        # "#FF9900",  # Neon Orange
                    ]
    def reset(self):
        self.image_path = None
        self.image = None
        self.optimal_k = None


    def set_image_path(self, image_path):
        """Sets the path to the image for processing."""
        self.image_path = image_path

    def get_image(self):
        """
        Reads and converts the 16-bit TIFF image to 8-bit.

        Returns:
            np.ndarray: 8-bit image
        """
        if self.image is None:
            image_16bit = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            self.image = self.convert_16bit_to_8bit(image_16bit)
        return self.image

    def convert_16bit_to_8bit(self, image_16bit):
        """Scales a 16-bit image to 8-bit."""
        return (255 * (image_16bit / 65535)).astype(np.uint8)

    def array_to_grayscale_img(self, arr):
        """
        Normalizes a single-channel float image to 8-bit grayscale.

        Args:
            arr (np.ndarray): Float32 array.

        Returns:
            np.ndarray: 8-bit normalized grayscale image.
        """
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-8:
            norm_arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm_arr = ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)
        return (norm_arr * 255).astype(np.uint8)

    def compute_surface_normals(self, depth):
        """
        Estimates surface normals from a depth map using Sobel filters.

        Args:
            depth (np.ndarray): Grayscale depth image.

        Returns:
            np.ndarray: Normal map of shape (H, W, 3).
        """
        for _ in range(5):
            depth = cv2.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=70)

        sobel_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5) / 4
        sobel_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5) / 4

        normal = np.zeros((*depth.shape, 3), dtype=np.float32)
        normal[..., 0] = -sobel_x
        normal[..., 1] = sobel_y
        normal[..., 2] = 1

        norm = np.sqrt(np.sum(np.square(normal), axis=2, keepdims=True))
        normal /= (norm + 1e-8)
        return normal

    def colorize_normals(self, normal):
        """
        Splits normal map into directional components and scales to [0, 1].

        Args:
            normal (np.ndarray): Normal map.

        Returns:
            Tuple[np.ndarray]: Colored normals, up, front, side components all in range [0, 1].
        """
        up = normal[..., 1]
        front = normal[..., 2]
        side = normal[..., 0]
        return (normal + 1) / 2, (up + 1) / 2, (front + 1) / 2, (side + 1) / 2
    
    def find_optimal_k_with_pca(self, normals, threshold=0.99):
        """
        Estimate optimal number of clusters using PCA.

        Args:
            normals (np.ndarray): Array of surface normals (N, 3).
            threshold (float): Cumulative explained variance ratio to reach.
            max_k (int): Maximum number of clusters to consider (e.g., 3).

        Returns:
            int: Estimated optimal number of clusters (1 ≤ k ≤ max_k).
        """
        self.logger.debug("Finding optimal k with PCA")
        # Reshape to (N, 3)
        reshaped_normals = normals.reshape(-1, 3)
        # Normalize normals to unit vectors
        normals = reshaped_normals / (np.linalg.norm(reshaped_normals, axis=1, keepdims=True) + 1e-6)

        # Check for low variance iamge
        if self.check_for_low_variance(reshaped_normals):
            return 1
        
        # Fit PCA
        pca = PCA(n_components=min(self.max_k, normals.shape[1]))
        pca.fit(normals)

        # Compute cumulative variance
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        # Select k based on cumulative variance threshold
        for i, val in enumerate(cum_var):
            if val >= threshold:
                self.logger.debug(f"Optimal k found: {i+1} | Explained Variance Ratio = {val}")
                return i + 1  # 1-based index
            
        self.logger.debug("No optimal k found. Using k=4.")
        return 4  # fallback to 4 if threshold not reached]
    
    def check_for_low_variance(self, normals, threshold=5.0):
        """ 
        Check variance in surface normals, if below threshold assign k=1.
        
        Args:
            normals (np.ndarray): Normalized surface normal vector array in shape (H*W, 3)
            threshold: (float): Threshold in degrees. If the standard deviation of the cosine similairity is 
                below threshold, this will indicate that a the image is flat and k should be equal to 1.

        Returns:
            (bool): True if variance in < threshold.
        """
        # Compute cosine similarity
        mean_normal = np.mean(normals, axis=0)
        mean_normal /= np.linalg.norm(mean_normal)
        cos_similarities = normals @ mean_normal  # shape (N,)
        angles = np.arccos(np.clip(cos_similarities, -1.0, 1.0))  # radians
        angular_std = np.rad2deg(np.std(angles))
        self.logger.debug(f"Image surface normal angular_std = {angular_std} degrees | Threshold = {threshold} degrees")
        if angular_std < threshold:
            return True
        else:
            return False 

    
    def find_optimal_k_with_elbow(self, normals, clustering_method, s=1):
        """
        Uses the Elbow method and KneeLocator to find the optimal number of clusters.
        
        Args:
            data (np.ndarray): Input array of shape (N, D).
            s (int): Elbow sensitivity - smaller values for S detect knees quicker, while larger values are more conservative. 
            
        Returns:
            int: Optimal number of clusters.
        """
        self.logger.debug("Finding optimal k with Elbow")
        inertias = []
        k_range = range(1, self.max_k + 1)
        reshaped_normals = normals.reshape(-1, 3)
        use_spherical = hasattr(self, 'cluster_with_spherical_kmeans') and callable(getattr(self, 'cluster_with_spherical_kmeans'))
    
        # Check for low variance iamge
        if self.check_for_low_variance(reshaped_normals):
            return 1

        for k in k_range:
            if clustering_method == "S_KMEANS":
                _, inertia = self.cluster_with_spherical_kmeans(normals, k=k)
            else:
                inertia, _, _ = cv2.kmeans(
                                        reshaped_normals,
                                        K=k,
                                        bestLabels=None,
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                        attempts=10,
                                        flags=cv2.KMEANS_PP_CENTERS
                                    )
            inertias.append(inertia)
        
        kneedle = KneeLocator(k_range, inertias, curve='convex', direction='decreasing',interp_method="polynomial", polynomial_degree=4, S=s)
        if kneedle.knee is not None:
            self.logger.debug(f"Optimal k found: {kneedle.knee}")
            # kneedle.plot_knee()
            # plt.title("Knee Detection")
            # plt.show()
            return kneedle.knee
        
        else:
            self.logger.debug("No optimal k found. Using k=3.")
            return 3
    
    def cluster_with_kmeans(self, colored_normals_smoothed, k=3):
        """
        Applies KMeans clustering to the normal map with adaptive k via PCA.

        Args:
            colored_normals_smoothed (np.ndarray): (H, W, 3) normal image after bilateral smoothing.
            normal_map (np.ndarray, optional): Original normal map (H, W, 3) for PCA-based k selection.

        Returns:
            np.ndarray: Label map of shape (H, W).
        """
        h, w, c = colored_normals_smoothed.shape
        reshaped = colored_normals_smoothed.reshape(h * w, c).astype(np.float32)

        _, labels, _ = cv2.kmeans(
                                reshaped,
                                K=k,
                                bestLabels=None,
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                attempts=10,
                                flags=cv2.KMEANS_PP_CENTERS
                            )
        return labels.flatten().reshape(h, w).astype(np.uint8)
    
    def orthogonal_centroids(self, X, k, eps=1e-6):
        """
        Select k orthogonal unit centroids for spherical KMeans from data X.
        
        Args:
            X (np.ndarray): Normalized data, shape (N, D)
            k (int): Number of centroids
            eps (float): Tolerance for re-orthogonalization

        Returns:
            np.ndarray: (k, D) centroids from X
        """
        n_samples, dim = X.shape
        centroids = []
        
        # Pick the first centroid randomly
        first = X[np.random.choice(n_samples)]
        centroids.append(first)

        # Generate additional centroids orthogonal to previous
        for _ in range(1, k):
            # Generate candidate directions orthogonal to existing centroids
            candidate = self.rng.random(dim)
            for prev in centroids:
                candidate -= np.dot(candidate, prev) * prev  # Gram-Schmidt orthogonalization
            norm = np.linalg.norm(candidate)
            if norm < eps:
                continue
            candidate /= norm

            # Find closest data point to the candidate vector
            sims = X @ candidate
            best_idx = np.argmax(sims)
            centroids.append(X[best_idx])

        return np.stack(centroids)

    def cluster_with_spherical_kmeans(self, normal_map, k=4, max_iter=100, tol=1e-4):
        """
        Perform spherical KMeans clustering using cosine similarity.

        Args:
            normal_map (np.ndarray): Input surface normal map of shape (n_samples, n_normals). Assumed to be normalized to unit vectors.
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence threshold on cosine similarity.

        Returns:
            labels (np.ndarray): Cluster labels for each data point.
            interia (np.ndarray): Inertia for each segment.
        """                          
        # Normalize
        X = normal_map.reshape(-1, 3).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Randomly initialize centroids from data
        # indices = np.random.choice(X.shape[0], k, replace=False)
        # centroids = X[indices]
        centroids = self.orthogonal_centroids(X, k)

        for n in range(max_iter):
            # Assign labels based on cosine similarity (dot product for normalized data)
            similarities = np.dot(X, centroids.T)  # (n_samples, k)
            labels = np.argmax(similarities, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    # Reinitialize empty cluster
                    new_centroids[i] = X[np.random.choice(X.shape[0])]
                else:
                    mean_vec = np.mean(cluster_points, axis=0) # get new centroid fro cluster mean
                    new_centroids[i] = mean_vec / (np.linalg.norm(mean_vec) + 1e-10) # norm centroid

            # Check for convergence
            delta = np.sum(1 - np.dot(new_centroids, centroids.T).diagonal())  # 1 - cosine similarity (cosine dist)
            # self.logger.debug(f"Iter {iteration}, change: {delta:.5f}")
            if delta < tol:
                break
            centroids = new_centroids

        # Compute  inertia
        dot_products = np.sum(X * centroids[labels], axis=1)  # sum of cosine similarity for all points to their centroids
        inertia = np.sum(1 - dot_products) # total cosine dist

        # Reshape labels in to 2D map
        label_map = labels.reshape(normal_map.shape[:2]).astype(np.uint8)
        return label_map, inertia

    
    def smooth_mask_open_close(self, label_map, open_ksize=5, close_ksize=7):
        """
        Apply morphological opening followed by closing to each cluster mask.
        Returns a cleaned label map.
        """
        cleaned_label_map = np.zeros_like(label_map)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))

        for cluster_id in range(self.optimal_k):
            mask = (label_map == cluster_id).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
            cleaned_label_map[mask > 0] = cluster_id

        return cleaned_label_map
    
    def contour_segments(self, label_map):
        """
        Find contours.

        Args:
            label_map (np.ndarray): Clustered label map.

        Returns:
            all_contours (Dict[int, List[np.ndarray]]): Contours per cluster ID.
            cleaned_label_map (np.ndarray): (Optional) Updated label map.
        """
        H, W = label_map.shape
        all_contours = {}
        for cluster_id in range(self.optimal_k):
            mask = (label_map == cluster_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours[cluster_id] = contours
        return all_contours

    def draw_contours(self, contours):
        """
        Draws contours on the original image.

        Args:
            contours (dict): Dictionary of contours by cluster ID.

        Returns:
            np.ndarray: Image with drawn contours.
        """
        image = self.image.copy()
        for idx, (cluster_id, contour) in enumerate(contours.items()):
            color = tuple(int(c * 255) for c in mcolors.to_rgb(self.colormap[idx % len(self.colormap)]))
            cv2.drawContours(image, contour, -1, color, 2)
        return image
    
    def visualize_labels(self, label_map):
        """
        Maps integer cluster labels to RGB colors.

        Args:
            label_map (np.ndarray): 2D array of labels.

        Returns:
            np.ndarray: Color visualization of the label map.
        """
        # color_list = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        base_cmap = plt.get_cmap('tab10' if self.optimal_k <= 10 else 'tab20', self.optimal_k)
        color_list = (base_cmap(range(self.optimal_k))[:, :3] * 255).astype(np.uint8) 
        vis = color_list[label_map % self.optimal_k]
        return vis

    def process_image(self, image_input, dynamic_k=None, clustering_method="KMEANS", neighbors=25, morph=True):
        """
        Full pipeline: from image input → depth → normals → clusters → contours.

        Args:
            image_input (str or np.ndarray): Either the path to the 16-bit input image (str),
                                            or an already loaded image (np.ndarray).
            dynamic_k (str): "PCA" for PCA based dynamic k, or "elbow" for elbow based kneedle approach. IF None, optimal is set to max_k.
            neighbors (int): Neighborhood for bilateral filter applied to surface normal map. 

        Returns:
            Tuple[np.ndarray, Dict[int, List[np.ndarray]]]:
                - label_map: 2D array of cluster labels.
                - contours: Contours per cluster.
        """
        self.reset()
        if isinstance(image_input, str):
            self.set_image_path(image_input)
            self.get_image()
            self.logger.debug(f"Image: {Path(image_input).stem}")
        elif isinstance(image_input, np.ndarray):
            self.image = self.convert_16bit_to_8bit(image_input)
        else:
            raise TypeError("image_input must be a file path (str) or an image array (np.ndarray)")

        if self.model is None:
            raise ValueError("No depth estimation model provided.")


        depth_unscaled = self.model.infer_image(self.image)  # float32 (H, W)
        depth_img = self.array_to_grayscale_img(depth_unscaled)
        normal_map = self.compute_surface_normals(depth_img)
        colored_normals, up, front, side = self.colorize_normals(normal_map)
        colored_normals_smoothed = cv2.bilateralFilter(colored_normals, d=neighbors, sigmaColor=75, sigmaSpace=70)

        if dynamic_k == 'PCA':
            self.optimal_k = self.find_optimal_k_with_pca(colored_normals_smoothed, threshold=0.99)
        elif dynamic_k == 'elbow':
            self.optimal_k = self.find_optimal_k_with_elbow(colored_normals_smoothed, clustering_method, s=2)
        else:
            reshaped_normals = colored_normals_smoothed.reshape(-1, 3)
            if self.check_for_low_variance(reshaped_normals):
                self.optimal_k = 1
            else:
                self.optimal_k = self.max_k

        if clustering_method == "KMEANS":
            self.logger.debug(f"Clustering with Standard Kmeans | k= {self.optimal_k}")
            label_map = self.cluster_with_kmeans(colored_normals_smoothed,  k=self.optimal_k)
        if clustering_method == "S_KMEANS":
            self.logger.debug(f"Clustering with Spherical Kmeans | k= {self.optimal_k}")
            label_map, _ = self.cluster_with_spherical_kmeans(colored_normals_smoothed, k=self.optimal_k)
        if morph:
            label_map = self.smooth_mask_open_close(label_map, open_ksize=7, close_ksize=9)
        contours = self.contour_segments(label_map)
        return label_map, contours

def run_test():

    import glob
    import os
    from pathlib import Path
    
    
    logging.basicConfig(
        format='%(name)s | %(levelname)s | %(message)s',
        level=logging.WARNING  # Default level for all loggers
    )
    logging.getLogger('NormalSegmenter').setLevel(logging.DEBUG)

    directory = "/Users/mikey/Library/Mobile Documents/com~apple~CloudDocs/Code/code_projects/isd-annotator/images/example_folder"
    file_paths = glob.glob(os.path.join(directory, "*.tif"))
    method = "pca_spherical_kmeans_with_morph"
    output_dir = Path(f"clustering_tests/{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in file_paths:
        # Import depth model
        encoder = 'vits'
        checkpoint_dir = 'src/depth_anything_v2/checkpoints'
        model_loader = DepthAnythingLoader(encoder=encoder, checkpoint_dir=checkpoint_dir)
        model = model_loader.get_model()

        # Run segmenter
        print("Testing")
        start = time.time()
        # image_path = "/Users/mikey/Library/Mobile Documents/com~apple~CloudDocs/Code/code_projects/isd-annotator/images/example_folder/wenqing_fan_040.tif"
        image_path = path
        segmenter = NormalSegmenter(model=model, max_k=3)
        label_map, contours = segmenter.process_image(image_path, dynamic_k=None, clustering_method="S_KMEANS", neighbors=25, morph=True)
        end = time.time()
        print("Total time: ", (end-start))
        # Show contours
        image = segmenter.draw_contours(contours)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis(False)
        # plt.show()

        #Show segmentation map
        viz = segmenter.visualize_labels(label_map)
        # plt.imshow(cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
        # plt.axis(False)
        # plt.show()

        combined = np.hstack((image, viz))
        plt.figure(figsize=(14,8))
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        plt.axis(False)
        plt.show()
        # plt.savefig(f"{output_dir}/{Path(image_path).stem}.png")
        # plt.close()

if __name__ == "__main__":
    run_test()

