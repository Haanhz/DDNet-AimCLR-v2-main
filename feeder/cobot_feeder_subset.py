import random
import numpy as np
import pickle, torch
from . import tools


class Feeder_single_subset(torch.utils.data.Dataset):
    """ Feeder for single inputs with data subset support """
    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, zero_z=False, 
                 data_ratio=1.0, random_seed=42):
        """
        Args:
            data_ratio: Ratio of training data to use (e.g., 0.8 for 80%)
            random_seed: Random seed for reproducible data subset selection
        """
        self.data_path = data_path
        self.label_path = label_path
        self.data_ratio = data_ratio
        self.random_seed = random_seed

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.zero_z = zero_z
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Apply data subset if ratio < 1.0
        if self.data_ratio < 1.0:
            self._apply_data_subset()

    def _apply_data_subset(self):
        """Apply data subset based on data_ratio"""
        total_samples = len(self.label)
        subset_size = int(total_samples * self.data_ratio)
        
        print(f"Applying data subset: {subset_size}/{total_samples} samples ({self.data_ratio*100:.1f}%)")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Create stratified subset (maintain class distribution)
        unique_labels = np.unique(self.label)
        subset_indices = []
        
        for label_class in unique_labels:
            # Get all indices for this class
            class_indices = np.where(np.array(self.label) == label_class)[0]
            class_count = len(class_indices)
            
            # Calculate subset size for this class
            class_subset_size = max(1, int(class_count * self.data_ratio))  # At least 1 sample per class
            
            # Randomly select subset for this class
            selected_indices = np.random.choice(class_indices, size=class_subset_size, replace=False)
            subset_indices.extend(selected_indices)
            
            print(f"  Class {label_class}: {class_subset_size}/{class_count} samples")
        
        # Convert to sorted array
        subset_indices = np.array(sorted(subset_indices))
        
        # Apply subset to data and labels
        if hasattr(self.data, '__getitem__'):  # For mmap arrays
            # Create index mapping for mmap data
            self.subset_indices = subset_indices
            self.original_data = self.data
            self.use_subset_indices = True
        else:
            # For regular arrays, directly subset
            self.data = self.data[subset_indices]
            self.use_subset_indices = False
        
        # Subset labels and sample names
        self.label = [self.label[i] for i in subset_indices]
        self.sample_name = [self.sample_name[i] for i in subset_indices]
        
        print(f"Data subset applied: {len(self.label)} samples selected")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        if hasattr(self, 'use_subset_indices') and self.use_subset_indices:
            # Use subset indices for mmap data
            actual_index = self.subset_indices[index]
            data_numpy = np.array(self.original_data[actual_index])
        else:
            # Direct indexing for regular arrays
            data_numpy = np.array(self.data[index])
            
        label = self.label[index]
        
        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        # Optionally zero out Z channel to treat data as 2D (C, T, V, M)
        if getattr(self, 'zero_z', False):
            data_numpy[:,:,2] = 0
        
        return data_numpy


class Feeder_triple_subset(torch.utils.data.Dataset):
    """ Feeder for triple inputs with data subset support """
    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, 
                 aug_method='12345', zero_z=False, data_ratio=1.0, random_seed=42):
        """
        Args:
            data_ratio: Ratio of training data to use (e.g., 0.8 for 80%)
            random_seed: Random seed for reproducible data subset selection
        """
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method
        self.data_ratio = data_ratio
        self.random_seed = random_seed

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.zero_z = zero_z

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # Apply data subset if ratio < 1.0
        if self.data_ratio < 1.0:
            self._apply_data_subset()

    def _apply_data_subset(self):
        """Apply data subset based on data_ratio (same logic as single feeder)"""
        total_samples = len(self.label)
        subset_size = int(total_samples * self.data_ratio)
        
        print(f"Applying data subset: {subset_size}/{total_samples} samples ({self.data_ratio*100:.1f}%)")
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # Create stratified subset (maintain class distribution)
        unique_labels = np.unique(self.label)
        subset_indices = []
        
        for label_class in unique_labels:
            # Get all indices for this class
            class_indices = np.where(np.array(self.label) == label_class)[0]
            class_count = len(class_indices)
            
            # Calculate subset size for this class
            class_subset_size = max(1, int(class_count * self.data_ratio))  # At least 1 sample per class
            
            # Randomly select subset for this class
            selected_indices = np.random.choice(class_indices, size=class_subset_size, replace=False)
            subset_indices.extend(selected_indices)
            
            print(f"  Class {label_class}: {class_subset_size}/{class_count} samples")
        
        # Convert to sorted array
        subset_indices = np.array(sorted(subset_indices))
        
        # Apply subset to data and labels
        if hasattr(self.data, '__getitem__'):  # For mmap arrays
            # Create index mapping for mmap data
            self.subset_indices = subset_indices
            self.original_data = self.data
            self.use_subset_indices = True
        else:
            # For regular arrays, directly subset
            self.data = self.data[subset_indices]
            self.use_subset_indices = False
        
        # Subset labels and sample names
        self.label = [self.label[i] for i in subset_indices]
        self.sample_name = [self.sample_name[i] for i in subset_indices]
        
        print(f"Data subset applied: {len(self.label)} samples selected")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        if hasattr(self, 'use_subset_indices') and self.use_subset_indices:
            # Use subset indices for mmap data
            actual_index = self.subset_indices[index]
            data_numpy = np.array(self.original_data[actual_index])
        else:
            # Direct indexing for regular arrays
            data_numpy = np.array(self.data[index])
            
        label = self.label[index]

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._aug(data_numpy)
        return [data1, data2, data3], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        # Optionally zero out Z channel to treat data as 2D (C, T, V, M)
        if getattr(self, 'zero_z', False):
            data_numpy[:,:,2] = 0
        
        return data_numpy

    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy, is_2d=getattr(self, 'zero_z', False))
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.aug_method:
            data_numpy = tools.random_scale_2d(data_numpy)
        if '8' in self.aug_method:
            data_numpy = tools.random_translate_2d(data_numpy)
        if '9' in self.aug_method:
            data_numpy = tools.random_joint_dropout_2d(data_numpy)
        if '0' in self.aug_method:
            data_numpy = tools.random_hand_emphasis_2d(data_numpy)
        
        # Optionally zero out Z channel to treat data as 2D (C, T, V, M)
        if getattr(self, 'zero_z', False):
            data_numpy[:,:,2] = 0
        
        return data_numpy