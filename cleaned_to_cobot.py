#!/usr/bin/env python3
import argparse
import pickle
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import copy
from scipy.signal import medfilt
import scipy.ndimage.interpolation as inter


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert cleaned raw action segments back to COBOT format for COBO_CLR training'
    )
    parser.add_argument('--cleaned_root', type=Path, required=True,
                       help='Root directory containing cleaned action folders')
    parser.add_argument('--out_root', type=Path, default=Path('cobot_dataset'),
                       help='Output directory for COBOT format data')
    parser.add_argument('--max_frame', type=int, default=60,
                       help='Maximum number of frames per sample')
    parser.add_argument('--resample', type=str, choices=['pad', 'center-crop', 'uniform-sample', 'zoom'],
                       default='zoom', help='Policy for segments longer than max_frame')
    parser.add_argument('--split_strategy', type=str, choices=['random', 'cross_subject'], 
                       default='cross_subject', help='Strategy for train/val split: random (80/20) or cross_subject (odd/even subjects)')
    return parser.parse_args()


def uniform_sample_indices(length: int, target: int) -> np.ndarray:
    """Uniform sampling for temporal resampling"""
    if length <= 0:
        return np.zeros((target,), dtype=np.int64)
    indices = np.linspace(0, max(0, length - 1), num=target)
    indices = np.clip(indices.round().astype(np.int64), 0, length - 1)
    return indices


def center_crop_indices(length: int, target: int) -> np.ndarray:
    """Center crop for temporal resampling"""
    if length <= target:
        return np.arange(length, dtype=np.int64)
    start_index = (length - target) // 2
    return np.arange(start_index, start_index + target, dtype=np.int64)


def zoom(p, target_l=60, joints_num=48, joints_dim=3):
    p_copy = copy.deepcopy(p)
    l = p_copy.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            # p_new[:, m, n] = medfilt(p_new[:, m, n], 3) # make no sense. p_new is empty.
            p_copy[:, m, n] = medfilt(p_copy[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p_copy[:, m, n], target_l/l)[:target_l]
    return p_new


def fit_to_length(data: np.ndarray, max_frame: int, policy: str) -> np.ndarray:
    """Resample data to fixed length"""
    length = data.shape[1]
    
    if length == max_frame:
        return data
    
    if length < max_frame:
        # Pad with last frame
        out = np.empty((3, max_frame, 48, 1), dtype=np.float32)
        out[:, :length] = data
        last = data[:, length - 1:length]
        out[:, length:] = last
        return out
    
    # length > max_frame
    if policy == 'uniform-sample':
        indices = uniform_sample_indices(length, max_frame)
        return data[:, indices]
    elif policy == 'center-crop':
        indices = center_crop_indices(length, max_frame)
        return data[:, indices]
    elif policy == 'pad':
        return data[:, :max_frame]
    else:
        raise ValueError(f'Unknown resample policy: {policy}')


def scan_cleaned_actions(cleaned_root: Path) -> List[Dict]:
    """Scan cleaned action folders and collect all segments"""
    samples = []
    
    for action_folder in tqdm(list(cleaned_root.iterdir()), desc='Scanning action folders'):
        if not action_folder.is_dir():
            continue
            
        action_name = action_folder.name
        
        # Scan all .npy files in this action folder
        for npy_file in action_folder.glob('*.npy'):
            try:
                # Load the cleaned segment
                segment = np.load(str(npy_file))
                
                # Validate shape
                if segment.ndim != 3 or segment.shape[1] != 48 or segment.shape[2] != 3:
                    logging.warning(f'Invalid shape {segment.shape} in {npy_file}, skipping')
                    continue
                
                # Extract action ID from filename: {subject_id}_{sample}_A{action_id}_{start_frame}_{stop_frame}
                filename = npy_file.stem
                try:
                    # Parse filename to extract subject ID and action ID
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        subject_id = int(parts[0])  # First part is subject ID
                        if parts[2].startswith('A'):
                            action_id = int(parts[2][1:])  # Remove 'A' prefix
                        else:
                            # Fallback: use folder name hash if parsing fails
                            action_id = hash(action_name) % 1000
                            logging.warning(f'Could not parse action ID from {filename}, using hash: {action_id}')
                    else:
                        # Fallback: use folder name hash if parsing fails
                        subject_id = hash(filename) % 100  # Generate a subject ID
                        action_id = hash(action_name) % 1000
                        logging.warning(f'Could not parse subject/action ID from {filename}, using hash: subject={subject_id}, action={action_id}')
                except (ValueError, IndexError) as e:
                    # Fallback: use folder name hash if parsing fails
                    subject_id = hash(filename) % 100  # Generate a subject ID
                    action_id = hash(action_name) % 1000
                    logging.warning(f'Failed to parse subject/action ID from {filename}: {e}, using hash: subject={subject_id}, action={action_id}')
                
                # Create sample info
                sample_name = f'{action_name}_{npy_file.stem}'
                
                samples.append({
                    'file_path': npy_file,
                    'action_id': action_id,
                    'action_name': action_name,
                    'subject_id': subject_id,
                    'sample_name': sample_name,
                    'length': segment.shape[0],
                    'segment': segment  # Keep in memory for now
                })
                
            except Exception as e:
                logging.error(f'Failed to load {npy_file}: {e}')
                continue
    
    return samples


def create_cross_subject_split(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train and validation sets based on subject ID (odd/even)"""
    train_samples = []
    val_samples = []
    
    for sample in samples:
        subject_id = sample['subject_id']
        if subject_id % 2 == 1:  # Odd subject IDs go to training
            train_samples.append(sample)
        else:  # Even subject IDs go to validation
            val_samples.append(sample)
    
    return train_samples, val_samples


def create_train_val_split(samples: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train and validation sets using random split"""
    np.random.seed(seed)
    
    # Group by action to ensure balanced split
    action_groups = {}
    for sample in samples:
        action_id = sample['action_id']
        if action_id not in action_groups:
            action_groups[action_id] = []
        action_groups[action_id].append(sample)
    
    train_samples = []
    val_samples = []
    
    for action_id, action_samples in action_groups.items():
        np.random.shuffle(action_samples)
        split_idx = int(len(action_samples) * train_ratio)
        
        train_samples.extend(action_samples[:split_idx])
        val_samples.extend(action_samples[split_idx:])
    
    return train_samples, val_samples


def write_ntu_data(samples: List[Dict], output_path: Path, max_frame: int, resample_policy: str) -> None:
    """Write samples to NTU format"""
    if not samples:
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data array
    data = np.empty((len(samples), max_frame, 48, 3), dtype=np.float32)
    names = []
    labels = []
    
    for i, sample in enumerate(tqdm(samples, desc='Converting to COBOT format')):
        all_data = sample['segment']
        
        # Resample to fixed length
        if resample_policy in ["uniform-sample", "center-crop", "pad"]:
            fitted_data = fit_to_length(all_data, max_frame, resample_policy)
        elif resample_policy == "zoom":
            fitted_data = zoom(all_data, max_frame, 48, 3)
        
        # Store
        data[i] = fitted_data
        names.append(sample['sample_name'])
        labels.append(sample['action_id'])
    
    # Save data
    np.save(str(output_path), data)
    
    # Save labels with correct naming convention
    if 'train_position' in str(output_path):
        label_path = output_path.parent / 'train_label.pkl'
    elif 'val_position' in str(output_path):
        label_path = output_path.parent / 'val_label.pkl'
    else:
        label_path = output_path.with_suffix('.pkl')
    
    with open(label_path, 'wb') as f:
        pickle.dump((names, labels), f)
    
    logging.info(f'Saved {len(samples)} samples to {output_path}')


def main():
    setup_logging()
    args = parse_args()
    
    # Scan cleaned actions
    samples = scan_cleaned_actions(args.cleaned_root)
    
    if not samples:
        logging.error('No valid samples found!')
        return
    
    logging.info(f'Found {len(samples)} samples across {len(set(s["action_id"] for s in samples))} actions')
    
    # Create train/val split
    if args.split_strategy == 'cross_subject':
        train_samples, val_samples = create_cross_subject_split(samples)
        logging.info(f'Using cross-subject split: odd subjects -> train, even subjects -> val')
    else: # random split
        train_samples, val_samples = create_train_val_split(
            samples, args.train_ratio, args.seed
        )
        logging.info(f'Using random split with {args.train_ratio:.1%} train ratio')
    
    logging.info(f'Train: {len(train_samples)} samples, Val: {len(val_samples)} samples')
    
    # Log subject distribution
    train_subjects = set(s['subject_id'] for s in train_samples)
    val_subjects = set(s['subject_id'] for s in val_samples)
    logging.info(f'Train subjects: {sorted(train_subjects)}')
    logging.info(f'Val subjects: {sorted(val_subjects)}')
    logging.info(f'Subject overlap: {len(train_subjects & val_subjects)}')
    
    # Create output directory
    if args.split_strategy == 'cross_subject':
        out_dir = args.out_root / 'xsub'  # Cross-subject evaluation
    else:
        out_dir = args.out_root / 'random'  # Random split
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train data
    train_path = out_dir / 'train_position.npy'
    write_ntu_data(train_samples, train_path, args.max_frame, args.resample)
    
    # Write val data
    val_path = out_dir / 'val_position.npy'
    write_ntu_data(val_samples, val_path, args.max_frame, args.resample)
    
    # Print action distribution
    train_actions = {}
    val_actions = {}
    
    for sample in train_samples:
        action_id = sample['action_id']
        train_actions[action_id] = train_actions.get(action_id, 0) + 1
    
    for sample in val_samples:
        action_id = sample['action_id']
        val_actions[action_id] = val_actions.get(action_id, 0) + 1
    
    logging.info('Action distribution:')
    for action_id in sorted(set(train_actions.keys()) | set(val_actions.keys())):
        train_count = train_actions.get(action_id, 0)
        val_count = val_actions.get(action_id, 0)
        logging.info(f'  Action {action_id}: train={train_count}, val={val_count}')
    
    # Log subject distribution per action for cross-subject split
    if args.split_strategy == 'cross_subject':
        logging.info('Subject distribution per action:')
        action_subjects = {}
        for sample in samples:
            action_id = sample['action_id']
            subject_id = sample['subject_id']
            if action_id not in action_subjects:
                action_subjects[action_id] = {'train': set(), 'val': set()}
            if subject_id % 2 == 1:
                action_subjects[action_id]['train'].add(subject_id)
            else:
                action_subjects[action_id]['val'].add(subject_id)
        
        for action_id in sorted(action_subjects.keys()):
            train_subs = sorted(action_subjects[action_id]['train'])
            val_subs = sorted(action_subjects[action_id]['val'])
            logging.info(f'  Action {action_id}: train_subjects={train_subs}, val_subjects={val_subs}')
    
    logging.info(f'Data ready for training in: {args.out_root}')


if __name__ == '__main__':
    main()