#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# ---------------- constants ----------------
RIGHT_HAND = list(range(0, 21))
LEFT_HAND  = list(range(21, 42))

def to_LJ3(arr: np.ndarray) -> np.ndarray:
    """Convert array to (L, 48, 3) format."""
    if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[-1] == 1:
        return np.transpose(arr, (1, 2, 0, 3))[..., 0].astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 5 and arr.shape[1] == 3 and arr.shape[-1] == 1:
        return np.transpose(arr[0], (1, 2, 0, 3))[..., 0].astype(np.float32)
    raise ValueError(f"Unexpected shape {arr.shape}")

def analyze_file_quality(file_path: Path, treat_zero_as_missing: bool = False, zero_eps: float = 0.0) -> dict:
    """Analyze quality of a single action file with comprehensive metrics."""
    try:
        # Load data
        data = np.load(file_path)
        data = to_LJ3(data)
        
        T, J, C = data.shape  # frames, joints, coordinates
        
        # Create mask for valid data
        if treat_zero_as_missing:
            valid_mask = (data != 0).all(axis=2) & np.isfinite(data).all(axis=2)
        else:
            valid_mask = np.isfinite(data).all(axis=2)
        
        # Basic quality metrics
        total_elements = T * J * C
        missing_elements = total_elements - valid_mask.sum() * C
        missing_percentage = (missing_elements / total_elements) * 100
        
        # Missing per frame
        missing_per_frame = (J - valid_mask.sum(axis=1)).mean()
        
        # High missing ratio (frames with >50% missing joints)
        high_missing_frames = (valid_mask.sum(axis=1) < J * 0.5).sum()
        high_missing_ratio = (high_missing_frames / T) * 100
        
        # Comprehensive gap analysis
        gaps_info = analyze_gaps(valid_mask, T)
        
        return {
            'file_path': str(file_path),
            'action': file_path.parent.name,
            'frames': T,
            'missing_percentage': missing_percentage,
            'missing_per_frame': missing_per_frame,
            'high_missing_ratio': high_missing_ratio,
            'max_gap_length': gaps_info['max_gap_length'],
            'max_gap_percentage': gaps_info['max_gap_percentage'],
            'total_gaps': gaps_info['total_gaps'],
            'avg_gap_length': gaps_info['avg_gap_length'],
            'gap_length_std': gaps_info['gap_length_std'],
            'long_gaps_count': gaps_info['long_gaps_count'],
            'quality_score': 100 - missing_percentage
        }
        
    except Exception as e:
        return {
            'file_path': str(file_path),
            'action': file_path.parent.name,
            'error': str(e),
            'quality_score': 0
        }

def analyze_gaps(valid_mask: np.ndarray, T: int) -> dict:
    """Analyze gaps in the valid mask."""
    J = valid_mask.shape[1]
    all_gaps = []
    
    # Analyze gaps for each joint
    for joint_idx in range(J):
        joint_valid = valid_mask[:, joint_idx]
        
        # Find gaps (contiguous False runs)
        gap_start = None
        for t in range(T):
            if not joint_valid[t] and gap_start is None:
                gap_start = t
            elif joint_valid[t] and gap_start is not None:
                gap_length = t - gap_start
                all_gaps.append(gap_length)
                gap_start = None
        
        # Handle gap at the end
        if gap_start is not None:
            gap_length = T - gap_start
            all_gaps.append(gap_length)
    
    if not all_gaps:
        return {
            'max_gap_length': 0,
            'max_gap_percentage': 0,
            'total_gaps': 0,
            'avg_gap_length': 0,
            'gap_length_std': 0,
            'long_gaps_count': 0
        }
    
    all_gaps = np.array(all_gaps)
    
    return {
        'max_gap_length': int(all_gaps.max()),
        'max_gap_percentage': (all_gaps.max() / T) * 100,
        'total_gaps': len(all_gaps),
        'avg_gap_length': float(all_gaps.mean()),
        'gap_length_std': float(all_gaps.std()),
        'long_gaps_count': int((all_gaps > T * 0.5).sum())  # Gaps > 50% of action length
    }

def main():
    parser = argparse.ArgumentParser(description='Clean low-quality COBOT action files (Comprehensive)')
    parser.add_argument('--actions_root', type=Path, default='output/actions_raw',
                       help='Root directory containing action subdirectories')
    parser.add_argument('--backup_dir', type=Path, default='output/actions_raw_backup',
                       help='Backup directory for removed files')
    parser.add_argument('--max_missing_pct', type=float, default=15.0,
                       help='Maximum allowed missing percentage (default: 15.0)')
    parser.add_argument('--max_missing_per_frame', type=float, default=8.0,
                       help='Maximum average missing joints per frame (default: 8.0)')
    parser.add_argument('--max_high_missing_ratio', type=float, default=30.0,
                       help='Maximum ratio of frames with >50%% missing joints (default: 30.0)')
    parser.add_argument('--max_gap_percentage', type=float, default=70.0,
                       help='Maximum gap length as percentage of action length (default: 70.0)')
    parser.add_argument('--max_avg_gap_length', type=float, default=10.0,
                       help='Maximum average gap length in frames (default: 10.0)')
    parser.add_argument('--max_long_gaps', type=int, default=5,
                       help='Maximum number of gaps >50%% of action length (default: 5)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually remove files (default: dry run)')
    parser.add_argument('--treat_zero_as_missing', action='store_true',
                       help='Treat zero coordinates as missing data')
    parser.add_argument('--zero_eps', type=float, default=0.0,
                       help='Epsilon for zero detection (default: 0.0)')
    
    args = parser.parse_args()
    
    print("🚀 COBOT Action Quality Cleaner (Comprehensive)")
    print("=" * 60)
    print(f"📁 Actions root: {args.actions_root}")
    print(f"💾 Backup directory: {args.backup_dir}")
    print(f"🔍 Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"📊 Quality thresholds:")
    print(f"   Max missing %: {args.max_missing_pct:.1f}%")
    print(f"   Max missing per frame: {args.max_missing_per_frame:.1f}")
    print(f"   Max high missing ratio: {args.max_high_missing_ratio:.1f}%")
    print(f"   Max gap percentage: {args.max_gap_percentage:.1f}%")
    print(f"   Max avg gap length: {args.max_avg_gap_length:.1f} frames")
    print(f"   Max long gaps: {args.max_long_gaps}")
    print()
    
    # Find all .npy files
    npy_files = list(args.actions_root.rglob("*.npy"))
    if not npy_files:
        print("❌ No .npy files found!")
        return
    
    print(f"🔍 Analyzing file quality...")
    
    # Analyze each file
    results = []
    for file_path in tqdm(npy_files, desc="Scanning actions"):
        result = analyze_file_quality(file_path, args.treat_zero_as_missing, args.zero_eps)
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter out files with errors
    if 'error' in df.columns:
        error_files = df[df['error'].notna()]
        df = df[df['error'].isna()].copy()
    else:
        error_files = pd.DataFrame()
    
    if df.empty:
        print("❌ No valid files found!")
        return
    
    # Apply quality thresholds
    low_quality_mask = (
        (df['missing_percentage'] > args.max_missing_pct) |
        (df['missing_per_frame'] > args.max_missing_per_frame) |
        (df['high_missing_ratio'] > args.max_high_missing_ratio) |
        (df['max_gap_percentage'] > args.max_gap_percentage) |
        (df['avg_gap_length'] > args.max_avg_gap_length) |
        (df['long_gaps_count'] > args.max_long_gaps)
    )
    
    high_quality_files = df[~low_quality_mask]
    low_quality_files = df[low_quality_mask]
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE QUALITY ANALYSIS REPORT")
    print("=" * 70)
    print()
    print(f"📁 Total files analyzed: {len(df):,}")
    print(f"✅ High quality files: {len(high_quality_files):,}")
    print(f"❌ Low quality files: {len(low_quality_files):,}")
    print(f"📊 Quality rate: {len(high_quality_files)/len(df)*100:.1f}%")
    
    if len(error_files) > 0:
        print(f"⚠️  Files with errors: {len(error_files):,}")
    
    # Show quality statistics
    if not df.empty:
        print(f"\n📈 Quality Statistics:")
        print(f"   Missing %: {df['missing_percentage'].mean():.2f}% ± {df['missing_percentage'].std():.2f}%")
        print(f"   Missing per frame: {df['missing_per_frame'].mean():.2f} ± {df['missing_per_frame'].std():.2f}")
        print(f"   High missing ratio: {df['high_missing_ratio'].mean():.2f}% ± {df['high_missing_ratio'].std():.2f}%")
        print(f"   Max gap %: {df['max_gap_percentage'].mean():.2f}% ± {df['max_gap_percentage'].std():.2f}%")
        print(f"   Avg gap length: {df['avg_gap_length'].mean():.2f} ± {df['avg_gap_length'].std():.2f} frames")
        print(f"   Long gaps count: {df['long_gaps_count'].mean():.2f} ± {df['long_gaps_count'].std():.2f}")
    
    # Show low quality files
    if len(low_quality_files) > 0:
        print(f"\n❌ LOW QUALITY FILES TO REMOVE:")
        print("-" * 60)
        
        # Group by reason for removal
        reasons = []
        for idx, row in low_quality_files.iterrows():
            reasons_for_file = []
            if row['missing_percentage'] > args.max_missing_pct:
                reasons_for_file.append(f"Missing: {row['missing_percentage']:.1f}%")
            if row['missing_per_frame'] > args.max_missing_per_frame:
                reasons_for_file.append(f"Per-frame: {row['missing_per_frame']:.1f}")
            if row['high_missing_ratio'] > args.max_high_missing_ratio:
                reasons_for_file.append(f"High missing: {row['high_missing_ratio']:.1f}%")
            if row['max_gap_percentage'] > args.max_gap_percentage:
                reasons_for_file.append(f"Gap: {row['max_gap_percentage']:.1f}%")
            if row['avg_gap_length'] > args.max_avg_gap_length:
                reasons_for_file.append(f"Avg gap: {row['avg_gap_length']:.1f}")
            if row['long_gaps_count'] > args.max_long_gaps:
                reasons_for_file.append(f"Long gaps: {row['long_gaps_count']}")
            
            reasons.append(f"  {row['action']}/{Path(row['file_path']).name}: {', '.join(reasons_for_file)}")
        
        for reason in reasons:
            print(reason)
    
    print("\n" + "=" * 70)
    
    # Save analysis
    df.to_csv('quality_analysis_comprehensive.csv', index=False)
    print(f"💾 Quality analysis saved to: quality_analysis_comprehensive.csv")
    
    if len(low_quality_files) == 0:
        print("✅ No low-quality files to remove!")
        print("\n🎉 Quality cleaning analysis complete!")
        print("💡 To actually remove files, run with --execute flag")
        return
    
    # Execute removal if requested
    if args.execute:
        print(f"\n🗑️  REMOVING LOW QUALITY FILES...")
        
        # Create backup directory
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        
        removed_count = 0
        for idx, row in low_quality_files.iterrows():
            file_path = Path(row['file_path'])
            backup_path = args.backup_dir / file_path.relative_to(args.actions_root)
            
            try:
                # Create backup directory structure
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file to backup
                shutil.move(str(file_path), str(backup_path))
                removed_count += 1
                
                print(f"  ✅ Moved: {file_path.name} → {backup_path}")
                
            except Exception as e:
                print(f"  ❌ Failed to move {file_path.name}: {e}")
        
        print(f"\n🎉 Successfully removed {removed_count} low-quality files!")
        print(f"💾 Files backed up to: {args.backup_dir}")
        
    else:
        print(f"\n💡 DRY RUN: {len(low_quality_files)} files would be removed")
        print("💡 To actually remove files, run with --execute flag")

if __name__ == "__main__":
    main()
