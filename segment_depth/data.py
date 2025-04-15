from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from sklearn.decomposition import PCA

class FeatureExtractionDataset(Dataset):
    def __init__(self, file_paths, n_samples_per_file=10, seed=None):
        """Dataset for feature extraction with random sampling
        
        Args:
            file_paths: List of paths to .npy files, where each file contains
                frames of shape (n_frames, height, width)
            n_samples_per_file: Number of frames to sample from each file
            seed: Random seed for reproducibility
        """
        self.file_paths = file_paths
        self.n_samples_per_file = n_samples_per_file
        self.rng = np.random.default_rng(seed)
        
        # Pre-sample file indices and frame indices
        self.sampled_indices = []
        for file_idx in range(len(file_paths)):
            # Load the file to get number of frames
            frames = np.load(file_paths[file_idx])
            n_frames = len(frames)
            
            # Sample frame indices for this file
            frame_indices = self.rng.choice(n_frames, 
                                          size=min(n_samples_per_file, n_frames),
                                          replace=False)
            
            # Store (file_idx, frame_idx) pairs
            self.sampled_indices.extend([(file_idx, frame_idx) for frame_idx in frame_indices])
        
    def __len__(self):
        return len(self.sampled_indices)
        
    def __getitem__(self, idx):
        file_idx, frame_idx = self.sampled_indices[idx]
        
        # Load the specific frame from the file
        frames = np.load(self.file_paths[file_idx])
        frame = frames[frame_idx]
        
        # Convert to RGB if grayscale (duplicate to 3 channels)
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], axis=2)
        elif frame.shape[2] == 1:
            frame = np.concatenate([frame, frame, frame], axis=2)
        
        # Ensure uint8 format for images (0-255)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
        return frame, file_idx

class BalancedVideoSampler:
    def __init__(self, 
                 model_name="google/mobilenet_v2_1.4_224", 
                 batch_size=32,
                 device=None,
                 n_clusters=30):
        """
        Creates a balanced dataset from video frames using feature extraction and clustering
        
        Args:
            model_name: HuggingFace model to use for feature extraction
            batch_size: Batch size for feature extraction
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            n_clusters: Number of clusters to create
        """
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Initialize clustering model
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                      batch_size=min(1000, n_clusters*3),
                                      random_state=42)
        
    def _extract_features(self, frames):
        """Extract features from frames using the model"""
        dataset = FeatureExtractionDataset(frames)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Process with feature extractor
                inputs = self.feature_extractor(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                outputs = self.model(**inputs)
                
                # Use pooled output or mean of last hidden state
                if hasattr(outputs, 'pooler_output'):
                    batch_features = outputs.pooler_output
                else:
                    batch_features = outputs.last_hidden_state.mean(dim=1)
                
                features.append(batch_features.cpu().numpy())
                
        return np.vstack(features)
    
    def _sample_frames(self, videos):
        """Sample frames from videos for initial processing
        
        Args:
            videos: numpy array of shape (n_videos, n_frames, height, width)
        
        Returns:
            Sampled frames and their indices
        """
        n_videos, n_frames, h, w = videos.shape
        
        # For extremely large datasets, we subsample first
        # Sample every Nth frame where N ensures we get ~10000 frames total
        sample_rate = max(1, int(n_videos * n_frames / 10000))
        
        all_sampled_frames = []
        all_indices = []
        
        for vid_idx in range(n_videos):
            for frame_idx in range(0, n_frames, sample_rate):
                all_sampled_frames.append(videos[vid_idx, frame_idx])
                all_indices.append((vid_idx, frame_idx))
                
        return np.array(all_sampled_frames), all_indices
    
    def create_balanced_dataset(self, videos, n_samples=1000, use_sampling=True):
        """
        Create a balanced dataset from videos
        
        Args:
            videos: numpy array of shape (n_videos, n_frames, height, width)
            n_samples: Number of samples to select
            use_sampling: Whether to pre-sample frames before feature extraction
            
        Returns:
            tuple: (selected_frames, indices)
                - selected_frames: numpy array of selected frames
                - indices: list of (video_idx, frame_idx) tuples for selected frames
        """
        # Optionally pre-sample to reduce computation
        if use_sampling:
            frames, indices = self._sample_frames(videos)
        else:
            # Flatten the videos into a list of frames
            n_videos, n_frames, h, w = videos.shape
            frames = videos.reshape(-1, h, w)
            indices = [(i//n_frames, i%n_frames) for i in range(n_videos * n_frames)]
        
        # Extract features
        features = self._extract_features(frames)
        
        # Perform clustering
        self.kmeans.fit(features)
        cluster_labels = self.kmeans.labels_
        
        # Select balanced samples from each cluster
        selected_indices = []
        samples_per_cluster = max(1, n_samples // self.n_clusters)
        
        for cluster_id in range(self.n_clusters):
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_members) == 0:
                continue
                
            # Take samples_per_cluster samples or all if less available
            selected = np.random.choice(
                cluster_members, 
                size=min(samples_per_cluster, len(cluster_members)),
                replace=False
            )
            selected_indices.extend(selected)
        
        # If we need more samples, take from largest clusters
        if len(selected_indices) < n_samples:
            cluster_sizes = np.bincount(cluster_labels)
            largest_clusters = np.argsort(cluster_sizes)[::-1]
            
            for cluster_id in largest_clusters:
                if len(selected_indices) >= n_samples:
                    break
                    
                cluster_members = np.where(cluster_labels == cluster_id)[0]
                already_selected = set(selected_indices)
                available = [i for i in cluster_members if i not in already_selected]
                
                if not available:
                    continue
                    
                n_to_select = min(len(available), n_samples - len(selected_indices))
                additional = np.random.choice(available, size=n_to_select, replace=False)
                selected_indices.extend(additional)
        
        # Cap at n_samples
        selected_indices = selected_indices[:n_samples]
        
        # Extract selected frames and their indices
        selected_frames = frames[selected_indices]
        selected_frame_indices = [indices[i] for i in selected_indices]
        
        return selected_frames, selected_frame_indices

# Custom dataset for SAM2 distillation
class SAM2Dataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.processor = processor
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # Generate random point prompts (you'd replace this with your actual point selection logic)
        h, w = image.height, image.width
        input_points = [[[np.random.randint(0, w), np.random.randint(0, h)]], [[1]]]  # x,y and label
        
        # Process inputs for teacher model
        inputs = self.processor(image, input_points=input_points, return_tensors="pt")
        
        return {"image": image, "inputs": inputs, "image_path": image_path}


