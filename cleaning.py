#!/usr/bin/env python3
"""
Test script for micro-gap cleaning using PCHIP + bone-length projection
Based on the recovery_joints implementation for COBOT dataset
"""

import argparse
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json

# No external imports needed - using improved fallback implementations
RECOVERY_AVAILABLE = False
print("📝 Using improved fallback implementation")


def build_mask_fallback(X: np.ndarray) -> np.ndarray:
    """Intelligent mask builder - detects gaps while allowing legitimate zeros"""
    # X: (T, 48, 3) - mask is True where coordinates are finite and not all-zero
    # This detects actual gaps (all-zero frames) while allowing legitimate zero coordinates
    
    # First, ensure all coordinates are finite
    finite_mask = np.isfinite(X).all(axis=2)
    
    # Then, detect frames where ALL coordinates are exactly zero (likely gaps)
    all_zero_mask = (X == 0).all(axis=2)
    
    # A joint is valid if it's finite AND not all-zero
    mask = finite_mask & ~all_zero_mask
    
    return mask


def _linear_extrap_edge(ts_obs, y_obs, ts_fill):
    """Linear extrapolation for edge gaps"""
    if len(ts_obs) < 2:
        return None
    i = slice(0, 2) if ts_fill[0] < ts_obs[0] else slice(-2, None)
    x2, y2 = ts_obs[i], y_obs[i]
    a = (y2[1] - y2[0]) / (x2[1] - x2[0])
    b = y2[0] - a * x2[0]
    return a * ts_fill + b

def pchip_fill_fallback(X: np.ndarray, M: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Improved PCHIP implementation with bounds checking and spatial constraints"""
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        print("❌ Scipy not available for PCHIP interpolation")
        return X.copy(), np.zeros((X.shape[0], X.shape[1]), dtype=bool)
    
    T, J, C = X.shape
    X_out = X.copy()
    imputed = np.zeros((T, J), dtype=bool)
    
    ts = np.arange(T)
    for j in range(J):
        obs = M[:, j]
        if obs.sum() < 4:  # need some support; otherwise skip
            continue
        
        # For each coordinate, prep interpolator over observed frames
        for c in range(C):
            y = X[:, j, c]
            if obs.sum() < 2:  # need at least 2 points for interpolation
                continue
                
            try:
                pchip = PchipInterpolator(ts[obs], y[obs], extrapolate=False)
                
                # Get observed value range for bounds checking
                y_obs = y[obs]
                y_min, y_max = y_obs.min(), y_obs.max()
                y_range = y_max - y_min
                y_margin = y_range * 0.2  # Allow 20% margin beyond observed range
                
                # fill only tiny gaps
                t = 0
                while t < T:
                    if not obs[t]:
                        s = t
                        while t < T and not obs[t]:
                            t += 1
                        e = t - 1
                        if (e - s + 1) <= max_gap:
                            if s > 0 and e < (T - 1):
                                # Interior gap - use PCHIP with bounds checking
                                try:
                                    y_interp = pchip(ts[s:e+1])
                                    
                                    # Apply bounds checking
                                    y_interp = np.clip(y_interp, y_min - y_margin, y_max + y_margin)
                                    
                                    # Check for unreasonable jumps (more than 2x observed range)
                                    if s > 0:  # Check jump from previous frame
                                        prev_val = X_out[s-1, j, c]
                                        max_jump = y_range * 2.0
                                        if abs(y_interp[0] - prev_val) > max_jump:
                                            # Use linear interpolation instead for this gap
                                            y_interp = np.linspace(prev_val, X_out[e+1, j, c], e-s+1)
                                    
                                    # Apply temporal smoothing to prevent sudden jumps
                                    if len(y_interp) > 1:
                                        # Simple moving average smoothing
                                        y_smooth = y_interp.copy()
                                        for i in range(1, len(y_interp)-1):
                                            y_smooth[i] = 0.25 * y_interp[i-1] + 0.5 * y_interp[i] + 0.25 * y_interp[i+1]
                                        y_interp = y_smooth
                                    
                                    X_out[s:e+1, j, c] = y_interp
                                    imputed[s:e+1, j] = True
                                except Exception:
                                    pass  # Skip if interpolation fails
                            else:
                                # Edge gap - use linear extrapolation
                                y_edge = _linear_extrap_edge(ts[obs], y[obs], ts[s:e+1])
                                if y_edge is not None:
                                    # Apply bounds checking
                                    y_edge = np.clip(y_edge, y_min - y_margin, y_max + y_margin)
                                    X_out[s:e+1, j, c] = y_edge
                                    imputed[s:e+1, j] = True
                    else:
                        t += 1
            except Exception:
                continue  # Skip this joint/coordinate if PCHIP fails
    
    return X_out, imputed


def compute_target_bone_lengths_fallback(X: np.ndarray, M: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Fallback bone length computation"""
    targets = {}
    
    # Define finger chains based on MediaPipe palm structure
    RIGHT_FINGERS = {
        "thumb":  [0, 1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }
    LEFT_FINGERS = {
        "thumb":  [21, 22, 23, 24, 25],
        "index":  [26, 27, 28, 29],
        "middle": [30, 31, 32, 33],
        "ring":   [34, 35, 36, 37],
        "pinky":  [38, 39, 40, 41],
    }
    
    finger_chains = list(RIGHT_FINGERS.values()) + list(LEFT_FINGERS.values())
    
    for chain in finger_chains:
        for a, b in zip(chain[:-1], chain[1:]):
            if a >= X.shape[1] or b >= X.shape[1]:
                continue
                
            # Compute bone lengths for frames where both joints are visible
            valid_frames = M[:, a] & M[:, b]
            if valid_frames.sum() < 5:  # need some support
                continue
                
            lengths = np.linalg.norm(X[valid_frames, b, :] - X[valid_frames, a, :], axis=1)
            # Use median for robustness
            median_length = np.median(lengths)
            if np.isfinite(median_length) and median_length > 0:
                targets[(a, b)] = float(median_length)
    
    return targets


def project_bone_lengths_fallback(X: np.ndarray, imputed: np.ndarray, 
                                target_lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Fallback bone length projection"""
    Xp = X.copy()
    
    def _project_child(parent_xyz, child_xyz, target_len, eps=1e-8):
        v = child_xyz - parent_xyz
        d = np.linalg.norm(v)
        if d < eps:
            # if direction collapsed, keep child on parent + small offset
            return parent_xyz + np.array([target_len, 0.0, 0.0])
        return parent_xyz + v * (target_len / d)
    
    # Define finger chains based on MediaPipe palm structure
    RIGHT_FINGERS = {
        "thumb":  [0, 1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }
    LEFT_FINGERS = {
        "thumb":  [21, 22, 23, 24, 25],
        "index":  [26, 27, 28, 29],
        "middle": [30, 31, 32, 33],
        "ring":   [34, 35, 36, 37],
        "pinky":  [38, 39, 40, 41],
    }
    
    # Process per hand, wrist→tips to ensure parents are ready
    finger_chains = list(RIGHT_FINGERS.values()) + list(LEFT_FINGERS.values())
    T = X.shape[0]
    
    for chain in finger_chains:
        for t in range(T):
            for a, b in zip(chain[:-1], chain[1:]):
                if (a, b) not in target_lengths:
                    continue
                # If child was imputed at t, project it using its parent at t
                if imputed[t, b]:
                    Xp[t, b, :] = _project_child(
                        Xp[t, a, :], Xp[t, b, :], target_lengths[(a, b)]
                    )
    
    return Xp

def crossfade_on_reappear(X_filled, X_raw, M, hand_idx, K=5):
    """Cross-fade on reappearance to kill 'blink' effect with spatial validation"""
    X_out = X_filled.copy()  # Start with filled data, not raw data
    T = M.shape[0]
    
    # Process each joint in the hand individually
    for joint_idx in hand_idx:
        if joint_idx >= M.shape[1]:
            continue
            
        # Find gaps for this specific joint
        joint_mask = M[:, joint_idx]
        t = 0
        
        while t < T:
            if not joint_mask[t]:  # Joint is missing
                s = t
                while t < T and not joint_mask[t]:
                    t += 1
                e = t - 1
                
                if t < T:  # Joint reappears
                    k = min(K, T - t)
                    w = np.linspace(1.0, 0.0, k, endpoint=False)[:, None]
                    sl = slice(t, t + k)
                    
                    # Get the filled and raw values for this joint
                    filled_vals = X_filled[sl, joint_idx, :]
                    raw_vals = X_raw[sl, joint_idx, :]
                    
                    # Check if raw values are valid (not all-zero) before cross-fading
                    raw_valid = (raw_vals != 0).any(axis=1)
                    
                    # Only cross-fade where raw data is valid
                    for i, is_valid in enumerate(raw_valid):
                        if is_valid:
                            # Cross-fade this specific joint (blend filled with raw for smooth transition)
                            X_out[sl.start + i, joint_idx, :] = (w[i] * filled_vals[i, :] + 
                                                                 (1 - w[i]) * raw_vals[i, :])
                        else:
                            # Keep filled data if raw data is invalid
                            X_out[sl.start + i, joint_idx, :] = filled_vals[i, :]
            else:
                t += 1
    
    return X_out


def clean_micro_gaps(X: np.ndarray, max_gap: int = 5) -> Tuple[np.ndarray, Dict]:
    """
    Clean micro-gaps using PCHIP + bone-length projection
    
    Args:
        X: Input skeleton data (T, 48, 3)
        max_gap: Maximum gap size to fill (≤5 frames recommended)
    
    Returns:
        X_cleaned: Cleaned skeleton data
        stats: Cleaning statistics
    """
    T, J, C = X.shape
    assert J == 48 and C == 3, f"Expected (T, 48, 3), got {X.shape}"
    
    print(f"  📊 Input shape: {X.shape}")
    
    # Step 1: Build mask of valid data
    if RECOVERY_AVAILABLE:
        M = build_mask(X)
    else:
        M = build_mask_fallback(X)
    
    valid_ratio = M.mean()
    print(f"  🔍 Valid data ratio: {valid_ratio:.2%}")
    
    # Debug: Show gap statistics
    gaps_found = 0
    total_gaps = 0
    for j in range(M.shape[1]):
        joint_mask = M[:, j]
        if joint_mask.sum() < joint_mask.shape[0]:  # Has gaps
            gaps_found += 1
            # Count consecutive gaps
            t = 0
            while t < joint_mask.shape[0]:
                if not joint_mask[t]:
                    s = t
                    while t < joint_mask.shape[0] and not joint_mask[t]:
                        t += 1
                    gap_size = t - s
                    if gap_size <= max_gap:
                        total_gaps += 1
                else:
                    t += 1
    
    print(f"  🔍 Joints with gaps: {gaps_found}/{M.shape[1]} (total fillable gaps: {total_gaps})")
    
    # Step 2: PCHIP fill only tiny gaps
    start_time = time.time()
    if RECOVERY_AVAILABLE:
        X1, imputed = pchip_fill_tiny_gaps(X, M, max_gap=max_gap)
    else:
        X1, imputed = pchip_fill_fallback(X, M, max_gap=max_gap)
    
    pchip_time = time.time() - start_time
    imputed_ratio = imputed.sum() / imputed.size
    print(f"  🎯 PCHIP filled {imputed.sum():,} joints ({imputed_ratio:.2%}) in {pchip_time:.3f}s")
    
    # Step 3: Compute target finger bone lengths from ORIGINAL observations only
    start_time = time.time()
    if RECOVERY_AVAILABLE:
        targets = compute_target_bone_lengths(X, M)  # Use original X and M, not imputed
    else:
        targets = compute_target_bone_lengths_fallback(X, M)  # Use original X and M, not imputed
    
    bone_time = time.time() - start_time
    print(f"  📏 Computed {len(targets)} target bone lengths from original observations in {bone_time:.3f}s")
    
    # Step 4: Project filled joints to enforce target lengths
    start_time = time.time()
    if RECOVERY_AVAILABLE:
        X2 = project_bone_lengths(X1, imputed, targets)
    else:
        X2 = project_bone_lengths_fallback(X1, imputed, targets)
    
    project_time = time.time() - start_time
    print(f"  🔧 Bone length projection completed in {project_time:.3f}s")
    
    # Step 5: Cross-fade on reappearance to kill "blink" effect
    start_time = time.time()
    RIGHT_HAND = list(range(0, 21))
    LEFT_HAND = list(range(21, 42))
    
    X3 = crossfade_on_reappear(X2, X.copy(), M, RIGHT_HAND, K=5)
    X3 = crossfade_on_reappear(X3, X.copy(), M, LEFT_HAND, K=5)
    
    crossfade_time = time.time() - start_time
    print(f"  🎭 Cross-fade on reappearance completed in {crossfade_time:.3f}s")
    
    # Step 6: Final validation - measure actual improvement
    # Count frames where we successfully filled gaps
    improvement_count = 0
    total_improvement = 0
    
    for j in range(M.shape[1]):
        original_mask = M[:, j]
        # Check if this joint had gaps that were filled
        if original_mask.sum() < original_mask.shape[0]:  # Had gaps
            # Count how many gaps were filled
            t = 0
            while t < original_mask.shape[0]:
                if not original_mask[t]:  # Gap found
                    s = t
                    while t < original_mask.shape[0] and not original_mask[t]:
                        t += 1
                    gap_size = t - s
                    if gap_size <= max_gap:
                        # Check if this gap was successfully filled
                        gap_filled = True
                        for gap_t in range(s, t):
                            if not np.isfinite(X3[gap_t, j, :]).all():
                                gap_filled = False
                                break
                        if gap_filled:
                            improvement_count += 1
                            total_improvement += gap_size
                else:
                    t += 1
    
    # Calculate improvement based on filled gaps
    if total_improvement > 0:
        improvement = total_improvement / (M.shape[0] * M.shape[1])  # Normalize by total frames*joints
        final_valid_ratio = valid_ratio + improvement
    else:
        improvement = 0.0
        final_valid_ratio = valid_ratio
    
    stats = {
        'input_shape': X.shape,
        'output_shape': X3.shape,
        'initial_valid_ratio': float(valid_ratio),
        'final_valid_ratio': float(final_valid_ratio),
        'improvement': float(improvement),
        'imputed_joints': int(imputed.sum()),
        'imputed_ratio': float(imputed_ratio),
        'target_bones': len(targets),
        'timing': {
            'pchip': float(pchip_time),
            'bone_lengths': float(bone_time),
            'projection': float(project_time),
            'crossfade': float(crossfade_time),
            'total': float(pchip_time + bone_time + project_time + crossfade_time)
        }
    }
    
    print(f"  ✅ Final valid ratio: {final_valid_ratio:.2%} (improvement: {improvement:+.2%})")
    
    # Debug: Check if data was actually modified
    if np.array_equal(X, X3):
        print(f"  ⚠️  WARNING: Output data is identical to input data!")
    else:
        print(f"  ✅ Data was successfully modified")
    
    return X3, stats


def process_single_file(npy_path: Path, output_dir: Path, max_gap: int = 5) -> Optional[Dict]:
    """Process a single .npy file"""
    try:
        print(f"\n🔄 Processing: {npy_path.name}")
        
        # Load data
        X = np.load(str(npy_path)).astype(np.float32)
        
        # Clean micro-gaps
        X_cleaned, stats = clean_micro_gaps(X, max_gap)
        
        # Save cleaned data
        output_path = output_dir / npy_path.name
        np.save(str(output_path), X_cleaned.astype(np.float32))
        
        # Add file info to stats
        stats['input_file'] = str(npy_path)
        stats['output_file'] = str(output_path)
        stats['file_size_mb'] = {
            'input': os.path.getsize(npy_path) / (1024 * 1024),
            'output': os.path.getsize(output_path) / (1024 * 1024)
        }
        
        print(f"  💾 Saved to: {output_path}")
        return stats
        
    except Exception as e:
        print(f"  ❌ Error processing {npy_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test micro-gap cleaning on COBOT dataset")
    parser.add_argument("--actions_root", type=Path, required=True, 
                       help="Path to actions root (folders per action with .npy files)")
    parser.add_argument("--output_dir", type=Path, default=Path("cleaned_micro_gaps"),
                       help="Output directory for cleaned files")
    parser.add_argument("--max_gap", type=int, default=5,
                       help="Maximum gap size to fill (≤5 frames recommended)")
    parser.add_argument("--limit_actions", type=int, default=None,
                       help="Limit number of actions to process (default: process ALL actions)")
    parser.add_argument("--per_action", type=int, default=None,
                       help="Limit number of samples per action (default: process ALL files per action)")
    parser.add_argument("--save_stats", type=Path, default=Path("micro_gap_cleaning_stats.json"),
                       help="Save detailed statistics to JSON file")
    
    args = parser.parse_args()
    
    print("🦴 MICRO-GAP CLEANING TEST")
    print("=" * 60)
    print(f"📁 Actions root: {args.actions_root}")
    print(f"📤 Output directory: {args.output_dir}")
    print(f"🎯 Max gap size: {args.max_gap} frames")
    print(f"📊 Recovery joints available: {RECOVERY_AVAILABLE}")
    print("=" * 60)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find action directories
    action_dirs = sorted([p for p in args.actions_root.iterdir() if p.is_dir()])
    total_actions = len(action_dirs)
    
    if args.limit_actions:
        action_dirs = action_dirs[:args.limit_actions]
        print(f"🎬 Found {total_actions} action directories (processing {len(action_dirs)} due to --limit_actions)")
    else:
        print(f"🎬 Found {total_actions} action directories (processing ALL)")
    
    all_stats = []
    total_processed = 0
    total_successful = 0
    
    start_time = time.time()
    
    for action_dir in action_dirs:
        print(f"\n📂 Processing action: {action_dir.name}")
        
        # Find .npy files in this action
        npy_files = sorted(action_dir.glob("*.npy"))
        total_files_in_action = len(npy_files)
        
        if args.per_action:
            npy_files = npy_files[:args.per_action]
            print(f"  📄 Found {total_files_in_action} .npy files (processing {len(npy_files)} due to --per_action)")
        else:
            print(f"  📄 Found {total_files_in_action} .npy files (processing ALL)")
        
        # Create action subdirectory in output
        action_output_dir = args.output_dir / action_dir.name
        action_output_dir.mkdir(exist_ok=True)
        
        for npy_file in npy_files:
            total_processed += 1
            
            # Process file
            stats = process_single_file(npy_file, action_output_dir, args.max_gap)
            
            if stats:
                total_successful += 1
                all_stats.append(stats)
            
            # Progress indicator
            if total_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed
                print(f"  📈 Progress: {total_processed} processed, {rate:.1f} files/sec")
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 MICRO-GAP CLEANING SUMMARY")
    print(f"{'='*60}")
    
    # Count total available files
    total_available_files = sum(len(list(p.glob("*.npy"))) for p in args.actions_root.iterdir() if p.is_dir())
    
    print(f"📁 Total actions available: {total_actions}")
    print(f"📄 Total files available: {total_available_files}")
    print(f"✅ Successfully processed: {total_successful}/{total_processed}")
    if args.limit_actions or args.per_action:
        print(f"🔒 Processing limited by: " + 
              (f"--limit_actions {args.limit_actions}" if args.limit_actions else "") +
              (" + " if args.limit_actions and args.per_action else "") +
              (f"--per_action {args.per_action}" if args.per_action else ""))
    else:
        print(f"🚀 Processing: ALL data (no limits)")
    
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"🚀 Average rate: {total_processed/total_time:.1f} files/sec")
    
    if all_stats:
        # Compute aggregate statistics
        improvements = [s['improvement'] for s in all_stats]
        imputed_ratios = [s['imputed_ratio'] for s in all_stats]
        
        print(f"\n📈 AGGREGATE STATISTICS:")
        print(f"  📊 Data improvement: {np.mean(improvements):+.2%} ± {np.std(improvements):.2%}")
        print(f"  🎯 Imputed joints: {np.mean(imputed_ratios):.2%} ± {np.std(imputed_ratios):.2%}")
        print(f"  📏 Target bones computed: {np.mean([s['target_bones'] for s in all_stats]):.1f}")
        
        # Save detailed statistics
        if args.save_stats:
            with open(args.save_stats, 'w') as f:
                json.dump({
                    'summary': {
                        'total_processed': total_processed,
                        'total_successful': total_successful,
                        'total_time': total_time,
                        'average_improvement': float(np.mean(improvements)),
                        'average_imputed_ratio': float(np.mean(imputed_ratios))
                    },
                    'files': all_stats
                }, f, indent=2)
            print(f"  💾 Detailed stats saved to: {args.save_stats}")
    
    print(f"\n🎉 Micro-gap cleaning test completed!")
    print(f"📁 Cleaned files saved to: {args.output_dir}")


def fit_to_length(data, target_length, method='uniform-sample'):
    """
    Resize sequence to target length
    
    Args:
        data: Input data (C, T, V, M) where C=channels, T=time, V=joints, M=persons
        target_length: Target temporal length
        method: Resampling method ('uniform-sample', 'center-crop', 'pad')
    
    Returns:
        resized_data: Data resized to target length
        original_length: Original temporal length
        method_used: Method actually used
    """
    C, T, V, M = data.shape
    
    if T == target_length:
        return data, T, 'no-change'
    
    if method == 'uniform-sample':
        # Uniform sampling across the temporal dimension
        indices = np.linspace(0, T-1, target_length, dtype=int)
        resized_data = data[:, indices, :, :]
        return resized_data, T, 'uniform-sample'
    
    elif method == 'center-crop':
        if T > target_length:
            # Center crop
            start = (T - target_length) // 2
            end = start + target_length
            resized_data = data[:, start:end, :, :]
        else:
            # Pad with zeros
            resized_data = np.zeros((C, target_length, V, M), dtype=data.dtype)
            start = (target_length - T) // 2
            resized_data[:, start:start+T, :, :] = data
        return resized_data, T, 'center-crop'
    
    elif method == 'pad':
        # Zero padding
        resized_data = np.zeros((C, target_length, V, M), dtype=data.dtype)
        copy_length = min(T, target_length)
        resized_data[:, :copy_length, :, :] = data[:, :copy_length, :, :]
        return resized_data, T, 'pad'
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    main()