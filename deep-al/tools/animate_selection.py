

import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from matplotlib.colors import ListedColormap

# Add parent directory to path to import pycls
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pycls.core.config import cfg
from pycls.datasets.data import Data

def get_experiment_dirs(base_path):
    """Find all experiment directories containing config.yaml."""
    exp_dirs = []
    if os.path.exists(os.path.join(base_path, 'config.yaml')):
        exp_dirs.append(base_path)
    else:
        for root, dirs, files in os.walk(base_path):
            if 'config.yaml' in files:
                exp_dirs.append(root)
    return exp_dirs

def load_experiment_config(exp_dir):
    """Load the experiment configuration."""
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(os.path.join(exp_dir, 'config.yaml'))
    return cfg

def load_dataset(cfg):
    """Load the dataset (points and labels)."""
    data_obj = Data(cfg)
    train_dataset, _ = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=False)
    
    if hasattr(train_dataset, 'tensors'):
        points = train_dataset.tensors[0].numpy()
        labels = train_dataset.tensors[1].numpy()
    elif hasattr(train_dataset, 'features') and hasattr(train_dataset, 'targets'):
        points = train_dataset.features.numpy() if isinstance(train_dataset.features, torch.Tensor) else train_dataset.features
        labels = train_dataset.targets.numpy() if isinstance(train_dataset.targets, torch.Tensor) else train_dataset.targets
    else:
        raise ValueError(f"Unknown dataset structure for {type(train_dataset)}")
    
    return points, labels

def load_coverages(exp_dir):
    """Load coverages.npy if it exists."""
    path = os.path.join(exp_dir, 'coverages.npy')
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception as e:
            print(f"Failed to load coverages: {e}")
    return None

def get_episode_dirs(exp_dir):
    """Get sorted list of episode directories."""
    return sorted(glob.glob(os.path.join(exp_dir, 'episode_*')), key=lambda x: int(os.path.basename(x).split('_')[-1]))

def get_relevant_indices(exp_dir, episode_dirs):
    """
    Determine relevant indices (L + U) from the experiment root (initial state).
    This aligns with BAYES_MISP initialization.
    """
    # Try loading from experiment root (Initial state)
    try:
        lSet_path = os.path.join(exp_dir, 'lSet.npy')
        uSet_path = os.path.join(exp_dir, 'uSet.npy')
        if os.path.exists(lSet_path) and os.path.exists(uSet_path):
            lSet_0 = np.load(lSet_path, allow_pickle=True).astype(int)
            uSet_0 = np.load(uSet_path, allow_pickle=True).astype(int)
            return np.concatenate([lSet_0, uSet_0]).astype(int)
    except Exception as e:
        print(f"Error loading initial sets from {exp_dir}: {e}")

    # Fallback to first episode (might be modified state!)
    if not episode_dirs:
        return None
    
    first_ep_dir = episode_dirs[0]
    try:
        lSet_0 = np.load(os.path.join(first_ep_dir, 'lSet.npy'), allow_pickle=True).astype(int)
        uSet_0 = np.load(os.path.join(first_ep_dir, 'uSet.npy'), allow_pickle=True).astype(int)
        return np.concatenate([lSet_0, uSet_0]).astype(int)
    except Exception as e:
        print(f"Error determining relevant indices from {first_ep_dir}: {e}")
        return None

def calculate_total_selections(exp_dir, episode_dirs):
    """Calculate total number of selections (Initial Pool + Active Sets)."""
    total = 0
    
    # Initial Pool
    initial_count = 0
    try:
        lSet_root = np.load(os.path.join(exp_dir, 'lSet.npy'), allow_pickle=True)
        initial_count = len(lSet_root)
    except Exception:
        pass
        
    if initial_count == 0 and episode_dirs:
        # Fallback: If root lSet is empty, check episode 0
        try:
            lSet_0 = np.load(os.path.join(episode_dirs[0], 'lSet.npy'), allow_pickle=True)
            initial_count = len(lSet_0)
        except Exception:
            pass
            
    total += initial_count
        
    # Active Sets
    if episode_dirs:
        for ep_dir in episode_dirs:
            try:
                p = os.path.join(ep_dir, 'activeSet.npy')
                if os.path.exists(p):
                    a_set = np.load(p, allow_pickle=True)
                    total += len(a_set)
            except Exception:
                pass
            
    return total

def build_frames_data(exp_dir, episode_dirs, coverages, points_relevant):
    """
    Construct the sequence of frames for the animation.
    Matches selections with coverage data.
    """
    frames = []
    global_selection_idx = 0
    
    # Check if we have enough coverage data
    total_selections = calculate_total_selections(exp_dir, episode_dirs)
    has_coverage = coverages is not None and len(coverages) >= total_selections
    
    if coverages is not None and not has_coverage:
        print(f"Warning: Coverages count ({len(coverages)}) < Total Selections ({total_selections}). Coverage map may be incomplete.")

    # 0. Initial Pool
    initial_active = np.array([], dtype=int)
    initial_lSet = np.array([], dtype=int)
    initial_uSet = np.array([], dtype=int)

    # Try loading from Root
    try:
        lSet_root = np.load(os.path.join(exp_dir, 'lSet.npy'), allow_pickle=True).astype(int)
        uSet_root = np.load(os.path.join(exp_dir, 'uSet.npy'), allow_pickle=True).astype(int)
    except Exception:
        lSet_root = np.array([], dtype=int)
        uSet_root = np.array([], dtype=int)

    if len(lSet_root) > 0:
        # Standard case: Initial pool was defined in root
        initial_active = lSet_root
        initial_lSet = np.array([], dtype=int)
        initial_uSet = np.concatenate([uSet_root, lSet_root])
    elif episode_dirs:
        # Fallback: Check if Episode 0 has an lSet (implied initial pool)
        try:
            lSet_ep0 = np.load(os.path.join(episode_dirs[0], 'lSet.npy'), allow_pickle=True).astype(int)
            if len(lSet_ep0) > 0:
                initial_active = lSet_ep0
                initial_lSet = np.array([], dtype=int)
                # If root lSet was empty, root uSet should contain everything
                if len(uSet_root) > 0:
                    initial_uSet = uSet_root
                else:
                    # Reconstruct if root uSet missing
                    uSet_ep0 = np.load(os.path.join(episode_dirs[0], 'uSet.npy'), allow_pickle=True).astype(int)
                    initial_uSet = np.concatenate([uSet_ep0, lSet_ep0])
        except Exception:
            pass

    if len(initial_active) > 0:
        for i, idx in enumerate(initial_active):
            frame = {
                'episode': 'Initial Pool',
                'step': i + 1,
                'total_steps': len(initial_active),
                'lSet': initial_lSet,
                'uSet': initial_uSet,
                'current_active': idx,
                'previously_selected': initial_active[:i]
            }
            
            if has_coverage and global_selection_idx < len(coverages):
                frame['coverage'] = coverages[global_selection_idx]
                global_selection_idx += 1
            elif points_relevant is not None:
                frame['coverage'] = np.zeros(len(points_relevant))
                
            frames.append(frame)

    for ep_dir in episode_dirs:
        ep_num = int(os.path.basename(ep_dir).split('_')[-1])
        
        # Load episode data
        try:
            lSet = np.load(os.path.join(ep_dir, 'lSet.npy'), allow_pickle=True).astype(int)
            uSet = np.load(os.path.join(ep_dir, 'uSet.npy'), allow_pickle=True).astype(int)
            active_path = os.path.join(ep_dir, 'activeSet.npy')
            activeSet = np.load(active_path, allow_pickle=True).astype(int) if os.path.exists(active_path) else np.array([], dtype=int)
        except Exception as e:
            print(f"Skipping {ep_dir} due to error: {e}")
            continue

        # Active Learning Selections
        if len(activeSet) == 0:
            # No selection in this episode (e.g. last episode)
            frames.append({
                'episode': ep_num,
                'step': 0,
                'total_steps': 0,
                'lSet': lSet,
                'uSet': uSet,
                'current_active': None,
                'previously_selected': np.array([], dtype=int),
                # Keep last coverage if available
                'coverage': coverages[global_selection_idx-1] if has_coverage and global_selection_idx > 0 else (np.zeros(len(points_relevant)) if points_relevant is not None else None)
            })
        else:
            for i, idx in enumerate(activeSet):
                frame = {
                    'episode': ep_num,
                    'step': i + 1,
                    'total_steps': len(activeSet),
                    'lSet': lSet,
                    'uSet': uSet,
                    'current_active': idx,
                    'previously_selected': activeSet[:i]
                }
                
                if has_coverage and global_selection_idx < len(coverages):
                    frame['coverage'] = coverages[global_selection_idx]
                    global_selection_idx += 1
                elif points_relevant is not None:
                    frame['coverage'] = np.zeros(len(points_relevant))
                    
                frames.append(frame)
                
    return frames

def create_selection_animation(frames_data, points, labels, exp_dir):
    """Create and save the selection animation."""
    print("Creating Selection Animation...")
    unique_classes = np.unique(labels)
    cmap = plt.get_cmap('tab10')
    class_colors = {cls: cmap(i % 10) for i, cls in enumerate(unique_classes)}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame_idx):
        ax.clear()
        data = frames_data[frame_idx]
        
        lSet = data['lSet']
        uSet = data['uSet']
        current_active = data['current_active']
        previously_selected = data['previously_selected']
        
        # Effective Sets
        effective_lSet = np.concatenate([lSet, previously_selected]) if len(previously_selected) > 0 else lSet
        
        exclude_from_uSet = list(previously_selected)
        if current_active is not None:
            exclude_from_uSet.append(current_active)
        exclude_mask = np.isin(uSet, exclude_from_uSet, invert=True)
        effective_uSet = uSet[exclude_mask]
        
        # Plot
        if len(effective_uSet) > 0:
            for cls in unique_classes:
                mask = labels[effective_uSet] == cls
                if np.any(mask):
                    ax.scatter(points[effective_uSet][mask, 0], points[effective_uSet][mask, 1],
                             color=class_colors[cls], alpha=0.15, s=20, label=f'Unlabeled {cls}')
                             
        if len(effective_lSet) > 0:
            for cls in unique_classes:
                mask = labels[effective_lSet] == cls
                if np.any(mask):
                    ax.scatter(points[effective_lSet][mask, 0], points[effective_lSet][mask, 1],
                             color=class_colors[cls], s=30, marker='o', label=f'Labeled {cls}')
                             
        if current_active is not None:
            cls = labels[current_active]
            ax.scatter(points[current_active, 0], points[current_active, 1],
                     color=class_colors[cls], edgecolors='black', s=150, marker='*', label='Selected')
                     
        ax.set_title(f"Episode {data['episode']} - Step {data.get('step', 0)}")
        ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
        ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=1000, repeat=False)
    save_animation(ani, exp_dir, 'selection_animation.mp4')
    plt.close(fig)

def create_coverage_animation(frames_data, points_relevant, points, exp_dir):
    """Create and save the coverage animation."""
    print("Creating Coverage Animation...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add colorbar (legend for coverage)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Coverage Probability')
    
    def update(frame_idx):
        ax.clear()
        data = frames_data[frame_idx]
        
        if 'coverage' in data and data['coverage'] is not None:
            cov = data['coverage']
            if len(cov) == len(points_relevant):
                ax.scatter(points_relevant[:, 0], points_relevant[:, 1], c=cov, cmap='viridis', s=20, vmin=0, vmax=1)
                ax.set_title(f"Coverage Map - Episode {data['episode']} Step {data.get('step', 0)}")
            else:
                ax.text(0.5, 0.5, f"Mismatch: Cov {len(cov)} vs Pts {len(points_relevant)}", ha='center')
        else:
            ax.text(0.5, 0.5, "No Coverage Data", ha='center')
            
        # Highlight selected point
        current_active = data.get('current_active')
        if current_active is not None:
            ax.scatter(points[current_active, 0], points[current_active, 1],
                     color='red', edgecolors='white', s=150, marker='*', label='Selected', linewidths=1.5)

        ax.set_xlim(points_relevant[:, 0].min() - 0.5, points_relevant[:, 0].max() + 0.5)
        ax.set_ylim(points_relevant[:, 1].min() - 0.5, points_relevant[:, 1].max() + 0.5)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=1000, repeat=False)
    save_animation(ani, exp_dir, 'coverage_animation.mp4')
    plt.close(fig)

def save_animation(ani, exp_dir, filename):
    save_path = os.path.join(exp_dir, filename)
    try:
        ani.save(save_path, writer='ffmpeg')
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to save {filename} as mp4: {e}. Trying gif...")
        try:
            ani.save(save_path.replace('.mp4', '.gif'), writer='pillow')
            print(f"Saved {filename.replace('.mp4', '.gif')}")
        except Exception as e2:
            print(f"Failed to save gif: {e2}")

def process_experiment(exp_dir, viz_mode='both'):
    print(f"Processing {exp_dir}...")
    
    try:
        cfg = load_experiment_config(exp_dir)
        points, labels = load_dataset(cfg)
        coverages = load_coverages(exp_dir)
        # EXP - Print the shape, average and std of the first row in coverages
        # print(f"Coverage shape: {coverages.shape}")
        
        # # print(f"Coverage shape: {coverages[59].shape}, mean: {np.mean(coverages[59])}, std: {np.std(coverages[59])}")
        # print(f"Coverage shape: {coverages[0].shape}, mean: {np.mean(coverages[0])}, std: {np.std(coverages[0])}")
        # sys.exit(0)
        # EXP
        episode_dirs = get_episode_dirs(exp_dir)
        
        relevant_indices = get_relevant_indices(exp_dir, episode_dirs)
        if relevant_indices is None:
            print("Could not determine relevant indices.")
            return
            
        points_relevant = points[relevant_indices]
        
        frames_data = build_frames_data(exp_dir, episode_dirs, coverages, points_relevant)
        
        if not frames_data:
            print("No frames generated.")
            return
            
        if viz_mode in ['both', 'selection']:
            create_selection_animation(frames_data, points, labels, exp_dir)
            
        if viz_mode in ['both', 'coverage']:
            create_coverage_animation(frames_data, points_relevant, points, exp_dir)
            
    except Exception as e:
        print(f"Error processing {exp_dir}: {e}")
        import traceback
        traceback.print_exc()

def animate_selection(base_path, viz_mode='both'):
    exp_dirs = get_experiment_dirs(base_path)
    print(f"Found {len(exp_dirs)} experiments.")
    
    for exp_dir in exp_dirs:
        process_experiment(exp_dir, viz_mode)

if __name__ == "__main__":
    # Example usage
    animate_selection(r"/cs/labs/daphna/dvir_mateless/lab_repo/daphna_lab/output/SCENARIO_A/resnet18/2025_12_4", viz_mode='both')