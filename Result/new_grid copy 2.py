import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define dataset and metrics
dataset = "credit"
metrics_list = ["t_rate", "model_shift", "acc"]

# Folder paths
folder_paths = {
    "Folder2": "New Experiments/topk_output/ftr",
    "Folder3": "New Experiments/topk_continual_static_output/ftr",
    "Folder4": "New Experiments/topk_continual_output/ftr",
    "Folder5": "New Experiments/diversek_output/ftr",
    "Folder6": "New Experiments/diversek_continual_output/ftr",
}

# Dictionary to store extracted data
data_dict = {
    folder: {metric: [] for metric in metrics_list} for folder in folder_paths
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
            if dataset_name != dataset:
                continue  # Skip files not in our dataset list

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Store values for each metric
            for metric in metrics_list:
                if metric in df.columns:
                    values = df[metric].astype(str).values  # Convert to string
                    cleaned_values = []

                    # Skip first 6 values if metric is "acc", otherwise skip 3
                    skip_count = 1 if metric in ["acc", "model_shift"] else 0
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

                    data_dict[folder_name][metric].extend(cleaned_values)

# Colors and labels
colors = {
    "Folder2": "#ff7f0e",
    "Folder3": "#2ca02c",
    "Folder4": "#d62728",
    "Folder5": "#9467bd",
    "Folder6": "#8c564b"
}
linestyles = {folder: "-" for folder in folder_paths}
labels = {
    "Folder2": "Top-k",
    "Folder3": "Top-k with static continual learning",
    "Folder4": "Top-k with DCL",
    "Folder5": "Fair Top-k",
    "Folder6": "Fair Top-k with DCL",
}

# Y-axis limits for each metric
y_limits = {
    "t_rate": (0, 1.4),
    "model_shift": (0, 5),
    "acc": (0.5, 1)
}

# Compute adaptive y-limits for "model_shift"
all_model_shift_values = []
for folder_name in folder_paths:
    all_model_shift_values.extend(data_dict[folder_name]["model_shift"])

if all_model_shift_values:
    min_shift, max_shift = min(all_model_shift_values), max(all_model_shift_values)
    margin = (max_shift - min_shift) * 0.1  # 10% margin
    y_limits["model_shift"] = (max(0, min_shift - margin), max_shift + margin)
else:
    y_limits["model_shift"] = (0, 10)  # Default fallback if no data exists

# Store legend handles for separate legend plot
legend_handles = []
legend_labels = []

# Create a single row with 3 columns for each metric
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)

for i, metric in enumerate(metrics_list):
    ax = axes[i]

    for folder_name in folder_paths:
        values = data_dict[folder_name][metric]
        if values:
            # Ensure x starts from 1 if metric is "acc" or "model_shift"
            x_values = np.arange(1, len(values) + 1) if metric in ["acc", "model_shift"] else np.arange(len(values))

            line, = ax.plot(x_values, values, linestyle=linestyles[folder_name], alpha=0.8,
                            color=colors[folder_name], linewidth=2, label=labels[folder_name])

            # Collect legend handles only once
            if labels[folder_name] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(labels[folder_name])

    # Set y-axis limit dynamically based on the metric
    if metric in y_limits:
        ax.set_ylim(y_limits[metric])

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("Time Steps", fontsize=16, fontweight="normal")
    
    # Title for each subplot
    if metric == "t_rate":
        ax.set_title("Test Acceptance Rate", fontsize=18, fontweight="normal")
    elif metric == "model_shift":
        ax.set_title("Model Shift", fontsize=18, fontweight="normal")
    elif metric == "acc":
        ax.set_title("Short-Term Accuracy", fontsize=18, fontweight="normal")

    ax.tick_params(axis='both', which='major', labelsize=14)

# Add vertical side title for dataset
fig.text(-0.05, 0.5, "Credit", fontsize=23, fontweight="normal", rotation=90, va="center", ha="center")

# Add a legend in the bottom-right corner
fig.legend(legend_handles, legend_labels, loc="lower right", fontsize=14, frameon=True, ncol=1)

# Save and show the combined plot
plt.savefig(f"Result/credit_comparison_1row3cols.png")
plt.show()
