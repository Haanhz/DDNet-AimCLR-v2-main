#!/usr/bin/env python3
import argparse
import pickle
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Segment COBOT continuous skeleton dataset per action, save (L,48,3) segments'
        )
    )
    parser.add_argument('--raw_root', type=Path, default=Path('pose_new_v2'))
    parser.add_argument('--ann_root', type=Path, default=Path('Annotation_v4'))
    parser.add_argument(
        '--ignored',
        type=Path,
        default=None,
        help='Optional path to a text file with one video_id per line to ignore',
    )
    parser.add_argument(
        '--export_actions_root',
        type=Path,
        default=None,
        help='If set, also export isolated samples grouped by action name folders to this root',
    )

    args = parser.parse_args()
    return args


def read_ignored_list(path: Optional[Path]) -> set:
    if path is None or not path.exists():
        return set()
    ignored = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                ignored.add(token)
    return ignored


def list_pose_files(pose_root: Path) -> List[Path]:
    if not pose_root.exists():
        logging.error(f'Pose root not found: {pose_root}')
        return []
    return sorted([p for p in pose_root.iterdir() if p.suffix == '.npy'])



def parse_pose_filename(stem: str) -> Optional[Tuple[int, str, str]]:
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    raw_subject = parts[0]
    digits = ''.join(ch for ch in raw_subject if ch.isdigit())
    if digits == '':
        return None
    try:
        subject_id = int(digits)
    except Exception:
        return None
    subject_name = parts[1]
    video_id = '_'.join(parts[2:])
    return subject_id, subject_name, video_id


def build_annotation_path(ann_root: Path, subject_id: int, subject_name: str, video_id: str) -> Path:
    # find CSV inside per-subject subfolder
    base = ann_root / 'Annotation_v4'
    scan_root = base if base.exists() else ann_root
    candidates = [f's{subject_id}_{subject_name}', f'{subject_id}_{subject_name}']
    for folder_name in candidates:
        folder = scan_root / folder_name
        if not (folder.exists() and folder.is_dir()):
            continue
        # Try matching filenames
        possibles = [
            folder / f'{folder_name}_{video_id}.csv',
        ]
        if folder_name.startswith('s'):
            possibles.append(folder / f'{folder_name[1:]}_{video_id}.csv')
        for p in possibles:
            if p.exists():
                return p
        # Fallback: any file ending with _<video_id>.csv
        try:
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == '.csv' and f.name.endswith(f'_{video_id}.csv'):
                    return f
        except Exception:
            pass
    # Default expected path (for logging)
    return scan_root / f's{subject_id}_{subject_name}' / f's{subject_id}_{subject_name}_{video_id}.csv'


def validate_and_warn_overlaps_gaps(actions: pd.DataFrame) -> None:
    if actions.empty:
        return
    actions_sorted = actions.sort_values('start').reset_index(drop=True)
    for i in range(len(actions_sorted) - 1):
        cur_end = int(actions_sorted.loc[i, 'stop'])
        next_start = int(actions_sorted.loc[i + 1, 'start'])
        if next_start <= cur_end:
            logging.warning('Detected overlapping actions in CSV (start <= previous stop).')
        elif next_start > cur_end + 1:
            logging.warning('Detected gap between actions in CSV (start > previous stop + 1).')



def scan_dataset(
    raw_root: Path,
    ann_root: Path,
    ignored_video_ids: set,
) -> Tuple[List[Dict], Dict[str, int]]:
    # Support both when raw_root is the base directory containing pose_new_v2/ and when it is pose_new_v2 itself
    pose_root_candidate = raw_root / 'pose_new_v2'
    pose_root = pose_root_candidate if pose_root_candidate.exists() else raw_root
    logging.info(f'Scanning pose root: {pose_root}')
    pose_files = list_pose_files(pose_root)

    samples: List[Dict] = []
    action_histogram: Dict[str, int] = {}

    for pose_path in tqdm(pose_files, desc='Scanning pose files'):
        parsed = parse_pose_filename(pose_path.stem)
        if parsed is None:
            logging.warning(f'Skipping unrecognized filename: {pose_path.name}')
            continue
        subject_id, subject_name, video_id = parsed

        if video_id in ignored_video_ids:
            continue

        csv_path = build_annotation_path(ann_root, subject_id, subject_name, video_id)
        if not csv_path.exists():
            logging.warning(f'Missing annotation CSV for {pose_path.name} -> {csv_path}')
            continue

        try:
            skel = np.load(str(pose_path), mmap_mode='r')
        except Exception as e:
            logging.error(f'Failed to load skeleton: {pose_path} ({e})')
            continue

        if skel.ndim != 3 or skel.shape[1] != 48 or skel.shape[2] != 3:
            logging.warning(f'Unexpected skeleton shape {skel.shape} in {pose_path.name}, skipping')
            continue

        total_frames = skel.shape[0]

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f'Failed to read CSV: {csv_path} ({e})')
            continue

        df_columns = {c.strip().lower(): c for c in df.columns}
        id_col = df_columns.get('id', 'ID')
        start_col = df_columns.get('start', 'start')
        stop_col = df_columns.get('stop', 'stop')
        label_col = df_columns.get('label')

        valid_count = 0
        error_count = 0

        for _, row in df.iterrows():
            try:
                action_id = int(row[id_col])
                start = int(row[start_col])
                end = int(row[stop_col])
            except Exception:
                continue

            if start > end:
                logging.warning(f'{pose_path.name}: Invalid segment start>end: {start}>{end}')
                error_count += 1
                continue
            if start < 0 or end >= total_frames:
                logging.warning(f'{pose_path.name}: Segment out of range [0,{total_frames-1}] vs [{start},{end}]')
                error_count += 1
                continue

            valid_count += 1
            length = end - start + 1
            sample_name = f'{subject_id}_{video_id}_A{action_id}_S{start}_E{end}'
            action_name = str(row[label_col]).strip() if label_col and label_col in df.columns else None

            samples.append({
                'pose_path': pose_path,
                'subject_id': subject_id,
                'subject_name': subject_name,
                'video_id': video_id,
                'csv_path': csv_path,
                'action_id': action_id,
                'action_name': action_name,
                'start': start,
                'end': end,
                'length': length,
                'sample_name': sample_name,
            })
            action_histogram[str(action_id)] = action_histogram.get(str(action_id), 0) + 1

        logging.info(f'{pose_path.name}: {valid_count} valid segments, {error_count} invalid segments')

    return samples, action_histogram



def main() -> None:
    setup_logging()
    args = parse_args()

    # Roots
    raw_root = args.raw_root
    ann_root = args.ann_root

    # Ignored list
    ignored_video_ids = read_ignored_list(args.ignored)
    if ignored_video_ids:
        logging.info(f'Loaded ignored list with {len(ignored_video_ids)} entries')

    # Determine subject split according to mode
    # First discover all pose files to infer subjects if needed
    pose_root_candidate = raw_root / 'pose_new_v2'
    pose_root = pose_root_candidate if pose_root_candidate.exists() else raw_root
    pose_files = list_pose_files(pose_root)

    # Scan and collect
    samples, action_hist = scan_dataset(
        raw_root=raw_root,
        ann_root=ann_root,
        ignored_video_ids = ignored_video_ids
    )

    if not samples:
        logging.error('No valid samples found. Nothing to do.')
        return

    # Summary before writing
    avg_len = np.mean([s['length'] for s in samples]) if samples else 0.0

    logging.info(f'Total samples: {len(samples)}')
    logging.info(f'Average segment length (frames): {avg_len:.2f}')
    logging.info(f'Unique actions: {len(action_hist)}')



    # Final report
    counts_per_action = ', '.join([f'{aid}:{cnt}' for aid, cnt in sorted(action_hist.items(), key=lambda x: int(x[0]))])
    logging.info(f'Per-action counts: {counts_per_action}')

    # Optional export grouped by action name
    if args.export_actions_root is not None:
        export_root: Path = args.export_actions_root
        export_root.mkdir(parents=True, exist_ok=True)
        # Group by pose to minimize reloads
        by_pose: Dict[Path, List[Dict]] = {}
        for s in samples:
            by_pose.setdefault(s['pose_path'], []).append(s)

        for pose_path, entries in tqdm(by_pose.items(), desc='Exporting per-action segments'):
            try:
                skel = np.load(str(pose_path))
            except Exception as e:
                logging.error(f'Failed to load skeleton: {pose_path} ({e})')
                continue

            for s in entries:
                start = s['start']
                end = s['end']
                act_name = s.get('action_name') or f'A{s["action_id"]}'
                safe_name = str(act_name).strip().replace('/', '-')
                dst_dir = export_root / safe_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                filename = f'{s["sample_name"]}.npy'
                dst_path = dst_dir / filename
                if dst_path.exists():
                    continue
                seg = skel[start : end + 1]
                np.save(str(dst_path), seg.astype(np.float32))

if __name__ == '__main__':
    main()