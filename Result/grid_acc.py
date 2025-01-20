import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the folders containing the CSV files
dataset = "UCI"
metrics = "failToRecourse"
folder_paths = {
    "Folder1": f"New Experiments/topk_output/{dataset}",
    "Folder2": f"New Experiments/diversek_continual_output/{dataset}"
}

# Dictionary to store extracted data separately for both folders
data_dict = {folder: {} for folder in folder_paths}

# Loop through both folders and collect data
for folder_name, folder_path in folder_paths.items():
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Ensure it's a CSV file
            parts = filename.split("_")
            if len(parts) < 6:
                continue  # Skip if filename doesn't match expected pattern

            try:
                recourse_num = float(parts[0])
                threshold = float(parts[1])
            except ValueError:
                continue  # Skip if parsing fails

            # Read the CSV file
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if metrics in df.columns:  # Ensure 'acc' column exists
                acc_values = df[metrics].values[3:]  # Skip first 3 values

                # Store data in a dictionary
                key = (recourse_num, threshold)
                if key not in data_dict[folder_name]:
                    data_dict[folder_name][key] = []
                data_dict[folder_name][key].extend(acc_values)

# Extract unique recourse and threshold values for subplot arrangement
recourse_values = sorted(set(k[0] for folder in data_dict.values() for k in folder.keys()))
threshold_values = sorted(set(k[1] for folder in data_dict.values() for k in folder.keys()))

# Set dynamic grid size based on available values
nrows, ncols = min(3, len(recourse_values)), min(3, len(threshold_values))

# Create a grid plot
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), constrained_layout=True)

# Improved color scheme
colors = {"Folder1": "#1f77b4", "Folder2": "#ff7f0e"}  # Deep blue & Orange
linestyles = {"Folder1": "-", "Folder2": "-"}
labels = {"Folder1": "Topk", "Folder2": "Fair-Topk with continual learning"}

# If there's only one row or column, make axes iterable
if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

# Store handles & labels for a single legend
handles = []
legend_labels = []

# Plot each (recourse_num, threshold) combination
for i, recourse_num in enumerate(recourse_values[:nrows]):
    for j, threshold in enumerate(threshold_values[:ncols]):
        ax = axes[i, j]
        key = (recourse_num, threshold)
        
        for folder_name in folder_paths.keys():
            if key in data_dict[folder_name]:
                acc_values = data_dict[folder_name][key]
                line, = ax.plot(acc_values, linestyle=linestyles[folder_name], alpha=0.8, 
                        color=colors[folder_name], linewidth=2, label=labels[folder_name])
                
                # Collect legend handles & labels only once
                if labels[folder_name] not in legend_labels:
                    handles.append(line)
                    legend_labels.append(labels[folder_name])

        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)  # Light dotted grid

# Add row titles (recourse numbers) on the left
for i, recourse_num in enumerate(recourse_values[:nrows]):
    axes[i, 0].set_ylabel(f"Recourse: {recourse_num}", fontsize=12, fontweight="bold")

# Add column titles (threshold values) on the top
for j, threshold in enumerate(threshold_values[:ncols]):
    axes[0, j].set_title(f"Threshold: {threshold}", fontsize=12, fontweight="bold")


# Create a single legend below all subplots
fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=12, frameon=True)

# Adjust layout to fit legend at the bottom
fig.subplots_adjust(bottom=0.15)

# Show the plot
plt.savefig(f"Result/grid_{metrics}_{dataset}.png")
