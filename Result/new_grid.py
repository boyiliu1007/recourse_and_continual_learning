import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define datasets and metrics
datasets = ["UCIcredit", "credit", "synthetic"]
metrics_list = ["t_rate", "model_shift", "acc"]

# Folder paths
folder_paths = {
    "Folder1": "New Experiments/topk_MLP_output",
    "Folder2": "New Experiments/topk_output"
}

# Dictionary to store extracted data
data_dict = {
    folder: {dataset: {metric: [] for metric in metrics_list} for dataset in datasets}
    for folder in folder_paths
}

# Load data
for folder_name, folder_path in folder_paths.items():
    if not os.path.exists(folder_path):
        continue  # Skip if folder doesn't exist

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            parts = filename.split("_")
            if len(parts) < 6:
                continue  # Skip invalid filenames

            dataset_name = parts[4]  # Extract dataset name
            if dataset_name not in datasets:
                continue  # Skip files not in our dataset list

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Store values for each metric
            for metric in metrics_list:
                if metric in df.columns:
                    values = df[metric].astype(str).values  # Convert to string
                    cleaned_values = []

                    # Skip first 6 values if metric is "acc", otherwise skip 3
                    skip_count = 1 if metric == "acc" else 0
                    values = values[skip_count:]

                    for val in values:
                        if "tensor" in val:  # Check if it's a tensor format
                            try:
                                num = float(val.replace("tensor(", "").replace(")", ""))  # Extract numeric value
                                cleaned_values.append(num)
                            except ValueError:
                                print(f"Skipping invalid tensor value: {val}")  # Debugging

                        else:
                            try:
                                cleaned_values.append(float(val))  # Convert normal numbers
                            except ValueError:
                                print(f"Skipping invalid value: {val}")

                    data_dict[folder_name][dataset_name][metric].extend(cleaned_values)
# Ensure data is loaded correctly
print("Data dictionary preview:", data_dict)

# Grid plot configuration
nrows, ncols = len(datasets), len(metrics_list)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10), constrained_layout=True)

# Colors and labels
colors = {"Folder1": "#1f77b4", "Folder2": "#ff7f0e"}  # Blue & Orange
linestyles = {"Folder1": "-", "Folder2": "-"}
labels = {"Folder1": "MLP", "Folder2": "vanilla"}

# Ensure axes is iterable when nrows or ncols = 1
if nrows == 1:
    axes = np.expand_dims(axes, axis=0)
if ncols == 1:
    axes = np.expand_dims(axes, axis=1)

# Store handles & labels for a single legend
handles = []
legend_labels = []
y_limits = {
    "t_rate": (0, 0.6),         # Adjust based on expected range
    "model_shift": (0, 2.5),  # Example range, adjust as needed
    "acc": (0.7, 1)
}

# Iterate through datasets and metrics
for i, dataset in enumerate(datasets):
    for j, metric in enumerate(metrics_list):
        ax = axes[i, j]

        for folder_name in folder_paths.keys():
            values = data_dict[folder_name][dataset][metric]
            if values:
                line, = ax.plot(values, linestyle=linestyles[folder_name], alpha=0.8, 
                                color=colors[folder_name], linewidth=2, label=labels[folder_name])

                # Collect legend handles & labels only once
                if labels[folder_name] not in legend_labels:
                    handles.append(line)
                    legend_labels.append(labels[folder_name])

        # Set y-axis limit dynamically based on the metric
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])

        ax.grid(True, linestyle=":", alpha=0.5)

# Add row titles (datasets)
for i, dataset in enumerate(datasets):
    axes[i, 0].set_ylabel(f"Dataset: {dataset}", fontsize=15, fontweight="normal")

# Add column titles (metrics)
for j, metric in enumerate(metrics_list):
    axes[0, j].set_title(f"Metric: {metric}", fontsize=15, fontweight="normal")

# Create a single legend below all subplots
fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=12, frameon=True)

# Adjust layout to fit legend
fig.subplots_adjust(bottom=0.15)

# Save the plot
plt.savefig(f"Result/grid_comparison.png")
plt.show()
