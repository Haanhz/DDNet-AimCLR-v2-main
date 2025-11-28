#!/usr/bin/env python3
import argparse
import pickle
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Load cleaned pose segments and save to pickle without resampling for supervised training'
    )

    parser.add_argument('--cleaned_root', type=Path, required=True,
                       help='Root directory containing cleaned action folders')

    parser.add_argument('--out_root', type=Path, default=Path('cobot_pickle'),
                       help='Output directory')

    parser.add_argument('--split_strategy', type=str,
                       choices=['random', 'cross_subject'],
                       default='cross_subject')

    parser.add_argument('--train_ratio', type=float, default=0.8) #for random split
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()
    

# --------------------------------
# 1. scan cleaned actions
# --------------------------------
def scan_cleaned_actions(cleaned_root: Path) -> List[Dict]:
    samples = []

    for action_folder in tqdm(list(cleaned_root.iterdir()),
                              desc='Scanning action folders'):
        if not action_folder.is_dir():
            continue

        action_name = action_folder.name

        for npy_file in action_folder.glob('*.npy'):
            try:
                segment = np.load(str(npy_file))

                if segment.ndim != 3 or segment.shape[1] != 48 or segment.shape[2] != 3:
                    logging.warning(f'Invalid shape {segment.shape} in {npy_file}')
                    continue

                filename = npy_file.stem
                parts = filename.split('_')

                try:
                    if len(parts) >= 3 and parts[2].startswith('A'):
                        subject_id = int(parts[0])
                        action_id = int(parts[2][1:])
                    else:
                        subject_id = hash(filename) % 100
                        action_id = hash(action_name) % 1000
                except:
                    subject_id = hash(filename) % 100
                    action_id = hash(action_name) % 1000

                samples.append({
                    'file_path': npy_file,
                    'action_id': action_id,
                    'action_name': action_name,
                    'subject_id': subject_id,
                    'sample_name': filename,
                    'length': segment.shape[0],
                    'segment': segment,
                })

            except Exception as e:
                logging.error(f'Failed to load {npy_file}: {e}')

    return samples


# --------------------------------
# 2. split data
# --------------------------------
def create_cross_subject_split(samples: List[Dict]):
    train_samples = []
    val_samples = []

    for sample in samples:
        if sample['subject_id'] % 2 == 1:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    return train_samples, val_samples


def create_random_split(samples: List[Dict], train_ratio, seed):
    np.random.seed(seed)
    np.random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    return samples[:split_idx], samples[split_idx:]


# --------------------------------
# 3. save pickle
# --------------------------------
def dump_raw_pickle(samples, output_path: Path):
    poses = []
    labels = []

    for sample in samples:
        poses.append(sample['segment'])
        labels.append(sample['action_id'])

    dataset = {
        "pose": poses,
        "label": labels
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info(f"Saved {len(poses)} samples to {output_path}")


# --------------------------------
# MAIN
# --------------------------------
def main():
    setup_logging()
    args = parse_args()

    samples = scan_cleaned_actions(args.cleaned_root)
    if not samples:
        logging.error('No valid samples found!')
        return

    logging.info(f'Found total {len(samples)} samples')

    if args.split_strategy == 'cross_subject':
        train_samples, val_samples = create_cross_subject_split(samples)
        logging.info("Using cross-subject split")
    else:
        train_samples, val_samples = create_random_split(
            samples, args.train_ratio, args.seed)
        logging.info("Using random split")

    logging.info(f'Train set: {len(train_samples)}')
    logging.info(f'Val set:   {len(val_samples)}')

    args.out_root.mkdir(parents=True, exist_ok=True)

    dump_raw_pickle(train_samples, args.out_root / "train_new_2.pickle")
    dump_raw_pickle(val_samples, args.out_root / "test_new_2.pickle")

    logging.info("Done!")


if __name__ == '__main__':
    main()
