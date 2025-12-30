import argparse
import pickle
import time

import numpy as np
import torch

from torchlight import import_class


def load_data(data_path, label_path, mmap=True):
    """Load COBOT skeleton data and labels from .npy / .pkl."""
    if mmap:
        data = np.load(data_path, mmap_mode="r")
    else:
        data = np.load(data_path)

    with open(label_path, "rb") as f:
        sample_names, labels = pickle.load(f)

    labels = np.array(labels)
    return data, labels, sample_names


def load_model(model_path, device):
    """
    Load AimCLR_v2_3views finetuned classifier, following the same config used in this repo.
    """
    AimCLR = import_class("net.aimclr_v2_3views_2.AimCLR_v2_3views")

    model = AimCLR(
        base_encoder="net.ddnet.DDNet_Original",
        pretrain=False,
        class_num=19,
        frame_l=60,
        joint_d=3,
        joint_n=48,
        filters=16,
        last_feture_dim=512,
        feat_d=1128,
    )

    checkpoint = torch.load(model_path, map_location=device)
    # Work with both plain state dicts and dicts with "model_state_dict" key
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def measure_inference_time(
    data,
    model,
    device,
    batch_size=1,
    warmup=True,
):
    """
    Measure average inference time per sample, using a warm-up loop
    and CUDA synchronization (if available), similar to the provided time.py.
    """
    num_samples = data.shape[0]
    times = []

    # Warm-up pass (not timed)
    if warmup:
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_data = data[i : i + batch_size]
                batch_tensor = torch.from_numpy(batch_data).float().to(device, non_blocking=True)
                _ = model(None, batch_tensor, stream="all")
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # Timed loop
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_data = data[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_data).float().to(device, non_blocking=True)

            start = time.time()
            _ = model(None, batch_tensor, stream="all")
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time_per_batch = float(np.mean(times))
    avg_time_per_sample = avg_time_per_batch / float(batch_size)
    return avg_time_per_sample


def main():
    parser = argparse.ArgumentParser(
        description="Measure COBOT AimCLR_v2_3views inference time per sample"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_cobot_clr_new/xsub/val_position.npy",
        help="Path to COBOT skeleton data (.npy)",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="data_cobot_clr_new/xsub/val_label.pkl",
        help="Path to COBOT labels (.pkl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="config/three-stream/linear/best_model.pt",
        help="Path to finetuned model checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use when measuring inference time",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Disable warm-up iterations before timing",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    data, labels, sample_names = load_data(args.data_path, args.label_path, mmap=True)
    print(f"#samples: {len(labels)}, data shape: {data.shape}")

    print("Loading model...")
    model = load_model(args.model_path, device)

    print("Measuring inference time...")
    avg_time_per_sample = measure_inference_time(
        data=data,
        model=model,
        device=device,
        batch_size=args.batch_size,
        warmup=not args.no_warmup,
    )

    print(f"Average inference time per sample: {avg_time_per_sample:.6f} s")
    print(f"Average inference time per sample: {avg_time_per_sample * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
