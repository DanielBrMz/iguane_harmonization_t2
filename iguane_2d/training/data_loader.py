"""
Data loading and preprocessing utilities
Handles loading pickle files and creating site-specific datasets
"""

import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple


def load_preprocessed_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed 4-slice data from pickle file
    
    Args:
        data_path: Path to pickle file containing preprocessed data
    
    Returns:
        Tuple of (images, gestational_age, sex, site) arrays
    """
    print(f"\nLoading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to appropriate types
    images = data['images'].astype(np.float32) / 255.0
    ga = data['gestational_age'].astype(np.float32)
    sex = data['sex'].astype(np.float32)
    site = data['site']  # Keep as string array
    
    print(f"  Images: shape={images.shape}, dtype={images.dtype}")
    print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  GA range: {ga.min():.1f} - {ga.max():.1f} weeks")
    print(f"  Sites: {np.unique(site)}")
    print(f"  Total slices: {len(images)}")
    
    return images, ga, sex, site


def create_site_datasets(
    images: np.ndarray,
    ga: np.ndarray,
    sex: np.ndarray,
    site: np.ndarray,
    reference_site: str = 'BCH_CHD'
) -> Tuple[Dict, str]:
    """
    Create separate datasets for each imaging site
    
    Args:
        images: Image array [N, H, W, C]
        ga: Gestational age array [N]
        sex: Sex array [N]
        site: Site labels [N]
        reference_site: Name of the reference site
    
    Returns:
        Tuple of (site_data_dict, reference_site_name)
    """
    site_data = {}
    unique_sites = np.unique(site)
    
    print(f"\nCreating datasets for {len(unique_sites)} sites:")
    
    for s in unique_sites:
        mask = site == s
        site_data[s] = {
            'images': images[mask],
            'ga': ga[mask],
            'sex': sex[mask],
            'n_slices': int(np.sum(mask))
        }
        print(f"  {s}: {site_data[s]['n_slices']} slices "
              f"(GA: {site_data[s]['ga'].min():.1f}-{site_data[s]['ga'].max():.1f}w)")
    
    # Validate reference site
    if reference_site not in site_data:
        print(f"\n  WARNING: Reference site '{reference_site}' not found!")
        print(f"  Available sites: {list(site_data.keys())}")
        reference_site = list(site_data.keys())[0]
        print(f"  Using '{reference_site}' as reference instead")
    else:
        print(f"\n  Reference site: {reference_site} ({site_data[reference_site]['n_slices']} slices)")
    
    return site_data, reference_site


class DataAugmenter:
    """Data augmentation for 2D brain slices"""
    
    def __init__(self, flip_prob=0.5, brightness_delta=0.05, 
                 contrast_lower=0.95, contrast_upper=1.05):
        """
        Initialize augmenter with probability thresholds
        
        Args:
            flip_prob: Probability of horizontal flip
            brightness_delta: Maximum brightness adjustment
            contrast_lower: Lower bound for contrast adjustment
            contrast_upper: Upper bound for contrast adjustment
        """
        self.flip_prob = flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
    
    @tf.function
    def augment(self, image, ga):
        """
        Apply random augmentations to image
        
        Args:
            image: Input image tensor [H, W, C]
            ga: Gestational age scalar
        
        Returns:
            Tuple of (augmented_image, ga)
        """
        # Random horizontal flip
        if tf.random.uniform(()) > (1.0 - self.flip_prob):
            image = tf.image.flip_left_right(image)
        
        # Random brightness adjustment
        if tf.random.uniform(()) > (1.0 - self.flip_prob):
            image = tf.image.random_brightness(image, max_delta=self.brightness_delta)
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Random contrast adjustment
        if tf.random.uniform(()) > (1.0 - self.flip_prob):
            image = tf.image.random_contrast(
                image, 
                lower=self.contrast_lower, 
                upper=self.contrast_upper
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, ga


def create_tf_dataset(
    images: np.ndarray,
    ga: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = False,
    augmenter: DataAugmenter = None
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset with optional augmentation
    
    Args:
        images: Image array [N, H, W, C]
        ga: Gestational age array [N]
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        augmenter: DataAugmenter instance (created if None and augment=True)
    
    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, ga))
    
    if shuffle:
        buffer_size = min(1000, len(images))
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    if augment:
        if augmenter is None:
            augmenter = DataAugmenter()
        dataset = dataset.map(
            augmenter.augment, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_all_datasets(
    site_data: Dict,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = True
) -> Dict[str, tf.data.Dataset]:
    """
    Create TensorFlow datasets for all sites
    
    Args:
        site_data: Dictionary of site data from create_site_datasets
        batch_size: Batch size for all datasets
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
    
    Returns:
        Dictionary mapping site names to tf.data.Dataset
    """
    augmenter = DataAugmenter() if augment else None
    site_datasets = {}
    
    print(f"\nCreating TensorFlow datasets:")
    print(f"  Batch size: {batch_size}")
    print(f"  Shuffle: {shuffle}")
    print(f"  Augmentation: {augment}")
    
    for site_name, site_dict in site_data.items():
        dataset = create_tf_dataset(
            images=site_dict['images'],
            ga=site_dict['ga'],
            batch_size=batch_size,
            shuffle=shuffle,
            augment=augment,
            augmenter=augmenter
        )
        site_datasets[site_name] = dataset
        n_batches = site_dict['n_slices'] // batch_size
        print(f"  {site_name}: {n_batches} batches/epoch")
    
    return site_datasets
