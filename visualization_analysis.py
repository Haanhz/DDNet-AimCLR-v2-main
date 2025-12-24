#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import pickle
import os
import argparse
from collections import defaultdict
import torch.nn.functional as F
from torchlight import import_class
import matplotlib as mpl
import torch.nn as nn


# Plot style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class COBOTVisualizationAnalyzer:
    """Visualization & analysis tool for COBOT dataset (t-SNE + correlation)."""
    def __init__(self, model_path, data_path, label_path, num_classes=19):
        self.model_path = model_path
        self.data_path = data_path
        self.label_path = label_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.action_names = [
            'Start', 'Stop', 'Slower', 'Faster', 'Done', 'FollowMe', 
            'Lift', 'Home', 'Interaction', 'Look', 'PickPart', 'DepositPart', 
            'Report', 'Ok', 'Again', 'Help', 'Joystick', 'Identification', 'Change'
        ]
        self.colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

    def load_model_and_data(self):
        print("Loading model and data...")
        self.data = np.load(self.data_path, mmap_mode='r')
        with open(self.label_path, 'rb') as f:
            self.sample_names, self.labels = pickle.load(f)
        print(f"Data shape: {self.data.shape}")
        print(f"Number of samples: {len(self.labels)}")
        print(f"Number of classes: {len(np.unique(self.labels))}")

        self.model = self.load_model()
        # ---- replace the classifier head with Identity on all three encoders ----
        for encoder_name in ['encoder_q', 'encoder_q_motion', 'encoder_q_bone']:
            encoder = getattr(self.model, encoder_name)
            linears = self._find_all_linear(encoder)

            if not linears:
                print(f"\n[DEBUG] No Linear layers found in {encoder_name}")
            else:
                print(f"\nReplacing {len(linears)} Linear layers in {encoder_name}:")
                for name, lin in linears:
                    print(f"  - {encoder_name}.{name}: Linear({lin.in_features} → {lin.out_features})")
                    self._set_module_by_name(encoder, name, nn.Identity())

        self.model.eval()

    def load_model(self):
        model_class = import_class('net.aimclr_v2_3views_2.AimCLR_v2_3views')
        model_args = {
            'base_encoder': 'net.ddnet.DDNet_Original',
            'pretrain': False,
            'class_num': 19,
            'frame_l': 60,
            'joint_d': 3,
            'joint_n': 48,
            'filters': 16,
            'last_feture_dim': 512,
            'feat_d': 1128
        }
        model = model_class(**model_args)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(self.device)
        return model

    def _find_all_linear(self, model: nn.Module):
        """
        Find all nn.Linear layers in a module.
        Returns list of (module_path, module_obj).
        """
        linears = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                linears.append((name, m))
        return linears

    
    def _set_module_by_name(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """
        Replace a submodule given its dotted path name (e.g., 'encoder.fc').
        """
        parts = module_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    def extract_features(self, batch_size=32, max_samples=None):
        print("Extracting features...")
        if max_samples is None:
            max_samples = len(self.labels)

        features, labels = [], []
        with torch.no_grad():
            for i in range(0, min(max_samples, len(self.labels)), batch_size):
                batch_end = min(i + batch_size, max_samples)
                batch_data = self.data[i:batch_end]
                batch_tensor = torch.from_numpy(batch_data).float().to(self.device)

                # three-stream fusion output as features
                batch_features = self.model(None, batch_tensor, stream='all')
                print("feat shape:", batch_features.shape)
                features.append(batch_features.cpu().numpy())
                labels.extend(self.labels[i:batch_end])

                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {batch_end}/{max_samples} samples")

        self.features = np.concatenate(features, axis=0)
        self.labels = np.array(labels)
        print(f"Extracted features shape: {self.features.shape}")
        return self.features, self.labels

    def compute_tsne(self, n_components=2, perplexity=30, early_exaggeration=12, max_iter=1000, random_state=42):
        """Compute t-SNE embedding (kept as in your code)."""
        print("Computing t-SNE embedding...")
        tsne = TSNE(
            n_components=n_components,
            early_exaggeration=early_exaggeration,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
            verbose=1
        )
        self.tsne_embedding = tsne.fit_transform(self.features)
        print(f"t-SNE embedding shape: {self.tsne_embedding.shape}")
        return self.tsne_embedding

    def compute_nmi(self, n_clusters=None):
        print("Computing NMI...")
        if n_clusters is None:
            n_clusters = self.num_classes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.features)
        nmi_score = normalized_mutual_info_score(self.labels, cluster_labels)
        print(f"NMI Score: {nmi_score:.4f}")
        return nmi_score, cluster_labels

    # --------- PLOTS (t-SNE and correlation matrix only) ---------

    def plot_tsne(self, figsize=(12, 10), save_path=None, nmi_score=None, compress = None, scale = None):
        """t-SNE with discrete class colors, clean legend, and optional NMI label."""
        
        emb = self.tsne_embedding.copy()
        if compress is not None and compress > 0 and compress < 1:
            unique_labels = np.unique(self.labels)
            centroids = {u: emb[self.labels == u].mean(axis=0) for u in unique_labels}
            emb = np.vstack([centroids[l] + (pt - centroids[l]) * compress
                    for pt, l in zip(emb, self.labels)])
                    
        if scale is not None and 0 < scale < 1:
            center = emb.mean(axis=0)
            emb = emb - center
            emb = emb * scale
            emb = emb + center

        print("Creating t-SNE visualization...")
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Discrete categorical colors (one fixed color per class)
        if self.num_classes <= 20:
            colors = plt.cm.get_cmap('tab20').colors[:self.num_classes]
        else:
            colors = list(plt.cm.get_cmap('tab20b').colors) + list(plt.cm.get_cmap('tab20c').colors)
            colors = colors[:self.num_classes]
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5, self.num_classes + 0.5, 1), cmap.N)

        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=self.labels,
            cmap=cmap, norm=norm,
            s=100, alpha=0.9, marker='o',
            linewidths=0, edgecolors='none'
        )

        # Legend with class names
        uniq = np.unique(self.labels)
        handles = [
            mpl.lines.Line2D([0],[0], marker='o', linestyle='', markersize=10,
                             markerfacecolor=cmap(i), markeredgecolor='black',
                             markeredgewidth=0.5,
                             label=(self.action_names[i] if i < len(self.action_names) else f'Class {i}'))
            for i in uniq
        ]
        leg = ax.legend(handles=handles, title="Actions", loc="upper left",
                        frameon=True, fancybox=True, shadow=False, framealpha=0.72)
        # Light legend background with a subtle black edge for clarity
        try:
            leg.get_frame().set_facecolor((0.95, 1.0, 1.0, 0.75))
            leg.get_frame().set_edgecolor('black')
        except Exception:
            pass

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.grid(True, linestyle=':', alpha=0.25)

        # NMI label in the top-right
        if nmi_score is not None:
            ax.text(
                0.98, 0.98, f"NMI: {nmi_score:.4f}",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=20, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85)
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to: {save_path}")
        plt.show()

    def plot_feature_correlation(self, save_path=None, sample_size=100):
        """Feature correlation matrix (first N features)."""
        print("Creating feature correlation matrix...")
        sample_size = int(min(sample_size, self.features.shape[1]))
        feature_corr = np.corrcoef(self.features[:, :sample_size].T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(feature_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'Feature Correlation Matrix (First {sample_size} features)')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        # Cleaner ticks when matrix is small
        if sample_size <= 30:
            ax.set_xticks(np.arange(sample_size))
            ax.set_yticks(np.arange(sample_size))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        plt.show()

    # -------------------------------------------------------------

    def generate_report(self, output_dir='./visualization_results'):
        print("Generating analysis...")
        os.makedirs(output_dir, exist_ok=True)

        self.extract_features()
        self.compute_tsne()

        nmi_score, _ = self.compute_nmi()

        # Only the two requested figures
        self.plot_tsne(save_path=os.path.join(output_dir, 'tsne_visualization.png'),
                       nmi_score=nmi_score)
        self.plot_feature_correlation(save_path=os.path.join(output_dir, 'feature_correlation.png'),
                                      sample_size=100)

        results = {
            'nmi_score': nmi_score,
            'num_samples': len(self.labels),
            'num_classes': len(np.unique(self.labels)),
            'feature_dim': self.features.shape[1],
            'class_distribution': dict(zip(*np.unique(self.labels, return_counts=True)))
        }

        with open(os.path.join(output_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("COBOT Dataset Analysis (t-SNE + Correlation)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"- Total samples: {len(self.labels)}\n")
            f.write(f"- Number of classes: {len(np.unique(self.labels))}\n")
            f.write(f"- Feature dimension: {self.features.shape[1]}\n")
            f.write(f"- NMI Score: {nmi_score:.4f}\n")

        print(f"Analysis complete! Results saved to: {output_dir}")
        print(f"NMI Score: {nmi_score:.4f}")
        return results

def main():
    parser = argparse.ArgumentParser(description='COBOT Dataset Visualization and Analysis')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file (.npy)')
    parser.add_argument('--label_path', type=str, required=True, help='Path to label file (.pkl)')
    parser.add_argument('--num_classes', type=int, default=19, help='Number of action classes')
    parser.add_argument('--output_dir', type=str, default='./visualization_results', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to process')
    args = parser.parse_args()
    
    analyzer = COBOTVisualizationAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
        label_path=args.label_path,
        num_classes=args.num_classes
    )
    
    analyzer.load_model_and_data()

    results = analyzer.generate_report(output_dir=args.output_dir)

    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"NMI Score: {results['nmi_score']:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
