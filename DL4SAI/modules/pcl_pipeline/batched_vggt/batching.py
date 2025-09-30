import cv2
import os
import shutil
import pickle

import numpy as np
from datetime import datetime

class Batching:
    """
    Split data input into batches
    """
    def __init__(self, data_path, image_path, verbose=False, use_cached=False, max_image_size=80, file_type=('.mp4', '.mov')):
        """
        Initialize the Batching class.

        Args:
            data_path (str): Directory containing video files.
            image_path (str): Directory where extracted images will be stored.
            verbose (bool): Print debugging info.
            use_cached (bool): Load batches from cache if available.
            max_image_size (int): Maximum number of frames per batch.
            file_type (tuple): Allowed video file type.
        """
        if not os.access(data_path, os.W_OK | os.X_OK):
            raise PermissionError(f"No write/execute access to: {data_path}")
        
        self.data_path = data_path
        self.verbose = verbose
        self.max_image_size = max_image_size
        self.max_per_video = max_image_size // 2
        self.file_type = file_type
        self.image_path = image_path

        self.cache_path = os.path.join(self.image_path, 'batches_cache.pkl')

        if use_cached and self._load_cached_batches():
            if self.verbose:
                print("Loaded batches from cache")
            return
        
        if os.path.exists(self.image_path):
            shutil.rmtree(self.image_path)
        os.makedirs(self.image_path)

        if not os.access(self.image_path, os.W_OK | os.X_OK):
            raise PermissionError(f"No write/execute access to: {image_path}")

        self.videos = self._load_video_files()
        self.frame_counts = self._get_video_frame_counts()
        self.batches, self.batches_size = self._create_batches()
        self._save_batches_to_cache()
        if self.verbose:
            print("New batches created and cached.")

    def _load_video_files(self):
        """
        Load and sort video file names from data directory
        """
        video_files = [f for f in os.listdir(self.data_path) if f.lower().endswith(self.file_type)]

        # Sort numerically based on time in filenames
        def extract_timestamp(filename):
            base = os.path.splitext(filename)[0]  # Remove extension

            if len(base) >= 15:  # Case 1: Timestamp pattern like PXL_20240608_153045123.mp4
                try:
                    timestamp_str = base.split('_')[1] + base.split('_')[2]
                    return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S%f")
                except (IndexError, ValueError):
                    # In case the filename does not match expected pattern
                    print(f"[WARNING] filename {filename} does not match expected pattern.")
                    return datetime.min
            
            elif len(base) <= 10:  # Case 2: Iterator pattern like IMG_8149.MOV
                if base.lower().startswith('img') and base[4:].isdigit():
                    return int(base[4:])
                else:
                    print(f"[WARNING] filename {filename} does not match expected pattern.")
                    return float('inf')
            
            else: 
                print(f"[WARNING] filename {filename} does not match expected pattern.")
                return float('inf')


        video_files.sort(key=extract_timestamp)

        full_paths = [os.path.join(self.data_path, f) for f in video_files]

        if self.verbose:
            print(f"Found {len(full_paths)} video files.")

        return full_paths
    
    def _get_video_frame_counts(self):
        frame_counts = {}
        for video_path in self.videos:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if self.verbose:
                    print(f"Failed to open {video_path}")
                frame_counts[video_path] = 0
                continue
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts[video_path] = count
            cap.release()
        return frame_counts
    
    def _uniform_frame_indices(self, total_frames, num_to_extract):
        if num_to_extract > total_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, num=num_to_extract, dtype=int).tolist()

    def is_blurry(self, image, threshold=10.0):
        """
        Detect blurry images
        Very sharp      Laplacian variance: 500+
        Slightly blurry Laplacian variance: 150-250
        Very blurry     Laplacian variance: <50
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var < threshold

    def _extract_frames_and_filter_blur(self, video_path, target_num, initial_overfetch=80):
        """
        Extract up to target_num sharp frames, uniformly across the video.
        Step 1: Overfetch more frames.
        Step 2: Filter all blurry ones.
        Step 3: Uniformly sample sharp frames from the filtered set.
        """
        total_frames = self.frame_counts[video_path]
        fetch_num = min(total_frames, initial_overfetch)

        candidate_indices = self._uniform_frame_indices(total_frames, fetch_num)
        cap = cv2.VideoCapture(video_path)
        sharp_frames_with_indices = []

        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            if not self.is_blurry(frame):
                sharp_frames_with_indices.append((idx, frame))
        cap.release()

        # Not enough sharp frames found
        if len(sharp_frames_with_indices) == 0:
            return [], []
        
        sharp_indices_all = [idx for idx, _ in sharp_frames_with_indices]

        # Uniformly pick target_num sharp frames from the valid ones
        num_available = len(sharp_frames_with_indices)
        if num_available <= target_num:
            selected = sharp_frames_with_indices
        else:
            # Use linspace over actual frame indices for true uniformity
            target_frame_positions = np.linspace(sharp_indices_all[0], sharp_indices_all[-1], num=target_num)
            
            # For each target position, find the closest available sharp frame
            selected = []
            used_indices = set()
            for target in target_frame_positions:
                closest_idx = min(
                    ((i, abs(i - target)) for i in sharp_indices_all if i not in used_indices),
                    key=lambda x: x[1]
                )[0]
                used_indices.add(closest_idx)
                selected.append(next((item for item in sharp_frames_with_indices if item[0] == closest_idx), None))

        sharp_indices = [idx for idx, _ in selected if idx is not None]
        sharp_frames = [frame for _, frame in selected if frame is not None]
        return sharp_frames, sharp_indices
    
    def _store_frames(self, video_path, frames, indices):
        """
        Save frames to disk and return a dictionary:
        { "video_dir": [timestamp_filename, ...] }
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.join(self.image_path, video_name)
        os.makedirs(video_dir, exist_ok=False)

        saved_paths = []
        for frame, idx in zip(frames, indices):
            fname = f"{idx}.png"
            out_path = os.path.join(video_dir, fname)
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        return saved_paths

    def _create_batches(self):
        """
        Create batches from consecutive videos.

        Returns:
            tuple:
                batches (list): Each batch is a list of image paths.
                batches_size (list): List of (n_left, n_right) frame counts.
        """
        n = len(self.videos)
        batches = []
        batches_size = []
        prev_right_video = None
        prev_right_frames = []
        prev_right_indices = []
        prev_right_images = []

        for i in range(n - 1):
            left_video = self.videos[i]
            right_video = self.videos[i + 1]

            # Extract frames for left video & store images
            if prev_right_video == left_video:
                # Reuse sharp frames from previous batch's right video to ensure overlap
                left_frames = prev_right_frames
                left_indices = prev_right_indices
                left_images = prev_right_images
            else:
                left_frames, left_indices = self._extract_frames_and_filter_blur(left_video, self.max_per_video)
                left_images = self._store_frames(left_video, left_frames, left_indices)

            # Extract frames for right video & store images
            right_frames, right_indices = self._extract_frames_and_filter_blur(right_video, self.max_per_video)
            right_images = self._store_frames(right_video, right_frames, right_indices)

            batches.append(left_images + right_images)
            n_left = len(left_frames)
            n_right = len(right_frames)
            batches_size.append((n_left, n_right))

            prev_right_video = right_video
            prev_right_frames = right_frames
            prev_right_indices = right_indices
            prev_right_images = right_images

            if self.verbose:
                print(f"Batch {i}: Left frames {n_left}, Right frames {n_right}, Total {n_left + n_right}")

        return batches, batches_size
    
    def _save_batches_to_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.batches, self.batches_size), f)
        except Exception as e:
            if self.verbose:
                print(f"Failed to save batch cache: {e}")

    
    def _load_cached_batches(self):
        if not os.path.exists(self.cache_path):
            if self.verbose:
                print("Batch cache file not found.")
            return False

        try:
            with open(self.cache_path, 'rb') as f:
                self.batches, self.batches_size = pickle.load(f)

            # Validate image paths
            for batch in self.batches:
                for img_path in batch:
                    if not os.path.exists(img_path):
                        if self.verbose:
                            print(f"Missing image: {img_path}")
                        return False

            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to load batch cache: {e}")
            return False


    def get_batches(self):
        return self.batches
    
    def get_batches_size(self):
        return self.batches_size
    
    def get_image_path(self):
        return self.image_path