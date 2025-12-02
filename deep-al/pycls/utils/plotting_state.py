"""
Unified plotting manager for training experiments.

Encapsulates all plotting concerns:
- State tracking (episodes, epochs, iterations, deltas)
- Visualization (creating plots)
- Persistence (saving arrays to disk)
"""

import os
import numpy as np


class PlottingManager:
    """Manages plotting state, visualization, and data persistence.
    
    This unified manager handles all plotting concerns so train_al.py can
    simply call methods like plot_episode(), save_data(), etc. without
    worrying about state management or function signatures.
    """
    
    def __init__(self):
        """Initialize all plotting state variables."""
        # Episode-level accuracy tracking
        self.episode_xvalues = []
        self.episode_yvalues = []
        
        # Epoch-level loss tracking (per episode)
        self.epoch_xvalues = []
        self.epoch_yvalues = []
        
        # Iteration-level loss tracking (per epoch)
        self.iteration_xvalues = []
        self.iteration_yvalues = []
        
        # DCoM-specific delta tracking
        self.delta_avg_lst = []
        self.delta_std_lst = []
    
    # ==================== State Management ====================
    
    def reset_epoch_plots(self):
        """Reset epoch and iteration plots for a new episode."""
        self.epoch_xvalues = []
        self.epoch_yvalues = []
        self.iteration_xvalues = []
        self.iteration_yvalues = []
    
    def reset_all(self):
        """Reset all plotting state."""
        self.episode_xvalues = []
        self.episode_yvalues = []
        self.epoch_xvalues = []
        self.epoch_yvalues = []
        self.iteration_xvalues = []
        self.iteration_yvalues = []
        self.delta_avg_lst = []
        self.delta_std_lst = []
    
    # ==================== Data Addition ====================
    
    def add_episode(self, x_val, y_val):
        """Add episode-level data point (episode number, test accuracy)."""
        self.episode_xvalues.append(x_val)
        self.episode_yvalues.append(y_val)
    
    def add_epoch(self, x_val, y_val):
        """Add epoch-level data point (epoch number, train loss)."""
        self.epoch_xvalues.append(x_val)
        self.epoch_yvalues.append(y_val)
    
    def add_iteration(self, x_val, y_val):
        """Add iteration-level data point (iteration number, loss)."""
        self.iteration_xvalues.append(x_val)
        self.iteration_yvalues.append(y_val)
    
    def add_delta_stats(self, avg, std):
        """Add DCoM delta statistics (average and std)."""
        self.delta_avg_lst.append(avg)
        self.delta_std_lst.append(std)
    
    # ==================== Data Retrieval ====================
    
    def get_episode_data(self):
        """Return episode-level plotting data."""
        return self.episode_xvalues, self.episode_yvalues
    
    def get_epoch_data(self):
        """Return epoch-level plotting data."""
        return self.epoch_xvalues, self.epoch_yvalues
    
    def get_iteration_data(self):
        """Return iteration-level plotting data."""
        return self.iteration_xvalues, self.iteration_yvalues
    
    def get_delta_stats(self):
        """Return DCoM delta statistics."""
        return self.delta_avg_lst, self.delta_std_lst
    
    # ==================== Plotting & Visualization ====================
    
    def plot_episode_results(self, dataset_name, out_dir, isDebug=False):
        """Plot episode-level results (test accuracy across episodes).
        
        Args:
            dataset_name: Name of the dataset (for title)
            out_dir: Directory to save the plot
            isDebug: Whether to print debug info
        """
        self._plot_arrays(
            self.episode_xvalues, self.episode_yvalues,
            x_name="Episodes", y_name="Test Accuracy",
            dataset_name=dataset_name, out_dir=out_dir, isDebug=isDebug
        )
    
    def plot_epoch_results(self, dataset_name, out_dir, isDebug=False):
        """Plot epoch-level results (training loss across epochs).
        
        Args:
            dataset_name: Name of the dataset (for title)
            out_dir: Directory to save the plot
            isDebug: Whether to print debug info
        """
        self._plot_arrays(
            self.epoch_xvalues, self.epoch_yvalues,
            x_name="Epochs", y_name="Loss",
            dataset_name=dataset_name, out_dir=out_dir, isDebug=isDebug
        )
    
    def plot_iteration_results(self, dataset_name, out_dir, isDebug=False):
        """Plot iteration-level results (loss across iterations).
        
        Args:
            dataset_name: Name of the dataset (for title)
            out_dir: Directory to save the plot
            isDebug: Whether to print debug info
        """
        self._plot_arrays(
            self.iteration_xvalues, self.iteration_yvalues,
            x_name="Iterations", y_name="Loss",
            dataset_name=dataset_name, out_dir=out_dir, isDebug=isDebug
        )
    
    def _plot_arrays(self, x_vals, y_vals, x_name, y_name, dataset_name, out_dir, isDebug=False):
        """Internal method to plot and save arrays to disk.
        
        Args:
            x_vals: X-axis values
            y_vals: Y-axis values
            x_name: Label for X-axis
            y_name: Label for Y-axis
            dataset_name: Name of the dataset (for title)
            out_dir: Directory to save the plot
            isDebug: Whether to print debug info
        """
        import matplotlib.pyplot as plt
        temp_name = "{}_vs_{}".format(x_name, y_name)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Dataset: {}; {}".format(dataset_name, temp_name))
        plt.plot(x_vals, y_vals)

        if isDebug:
            print("plot_saved at : {}".format(os.path.join(out_dir, temp_name + '.png')))

        plt.savefig(os.path.join(out_dir, temp_name + ".png"))
        plt.close()
    
    # ==================== Data Persistence ====================
    
    def save_episode_data(self, out_dir, isDebug=True):
        """Save episode-level data to disk.
        
        Args:
            out_dir: Directory to save files
            isDebug: Whether to print debug info
        """
        self._save_plot_values(
            [self.episode_xvalues, self.episode_yvalues],
            ["plot_episode_xvalues", "plot_episode_yvalues"],
            out_dir=out_dir, isDebug=isDebug
        )
    
    def save_epoch_data(self, out_dir, isDebug=True):
        """Save epoch-level data to disk.
        
        Args:
            out_dir: Directory to save files
            isDebug: Whether to print debug info
        """
        self._save_plot_values(
            [self.epoch_xvalues, self.epoch_yvalues],
            ["plot_epoch_xvalues", "plot_epoch_yvalues"],
            out_dir=out_dir, isDebug=isDebug
        )
    
    def save_iteration_data(self, out_dir, isDebug=True):
        """Save iteration-level data to disk.
        
        Args:
            out_dir: Directory to save files
            isDebug: Whether to print debug info
        """
        self._save_plot_values(
            [self.iteration_xvalues, self.iteration_yvalues],
            ["plot_it_x_values", "plot_it_y_values"],
            out_dir=out_dir, isDebug=isDebug
        )
    
    def save_delta_stats(self, out_dir, isDebug=True):
        """Save DCoM delta statistics to disk.
        
        Args:
            out_dir: Directory to save files
            isDebug: Whether to print debug info
        """
        self._save_plot_values(
            [self.delta_avg_lst, self.delta_std_lst],
            ["delta_avg_lst", "delta_std_lst"],
            out_dir=out_dir, isDebug=isDebug
        )
    
    def save_all_data(self, out_dir, isDebug=True):
        """Save all plotting data to disk.
        
        Args:
            out_dir: Directory to save files
            isDebug: Whether to print debug info
        """
        self.save_episode_data(out_dir, isDebug=isDebug)
        self.save_epoch_data(out_dir, isDebug=isDebug)
        self.save_iteration_data(out_dir, isDebug=isDebug)
        self.save_delta_stats(out_dir, isDebug=isDebug)
    
    def _save_plot_values(self, temp_arrays, temp_names, out_dir, saveInTextFormat=False, isDebug=True):
        """Internal method to save arrays to disk in numpy or text format.
        
        Args:
            temp_arrays: List of arrays to save
            temp_names: List of names for each array
            out_dir: Directory to save files
            saveInTextFormat: Whether to save as .txt or .npy
            isDebug: Whether to print debug info
        """
        for i in range(len(temp_arrays)):
            temp_arrays[i] = np.array(temp_arrays[i])
            temp_dir = out_dir
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            if saveInTextFormat:
                np.savetxt(temp_dir + '/' + temp_names[i] + ".txt", temp_arrays[i], fmt="%d")
            else:
                np.save(temp_dir + '/' + temp_names[i] + ".npy", temp_arrays[i])


# Legacy function aliases for backwards compatibility
def plot_arrays(x_vals, y_vals, x_name, y_name, dataset_name, out_dir, isDebug=False):
    """Legacy function - use PlottingManager.plot_episode_results() or similar instead."""
    manager = PlottingManager()
    manager._plot_arrays(x_vals, y_vals, x_name, y_name, dataset_name, out_dir, isDebug)


def save_plot_values(temp_arrays, temp_names, out_dir, isParallel=True, saveInTextFormat=False, isDebug=True):
    """Legacy function - use PlottingManager.save_*_data() methods instead."""
    manager = PlottingManager()
    manager._save_plot_values(temp_arrays, temp_names, out_dir, saveInTextFormat, isDebug)

