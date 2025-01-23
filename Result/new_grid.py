import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define datasets and metrics
datasets = ["UCIcredit", "credit", "synthetic"]
metrics_list = ["t_rate", "model_shift", "acc"]

# Folder paths
folder_paths = {
    "Folder2": "New Experiments/diversek_output/five",
    "Folder3": "New Experiments/topk_continual_static_output/five",
    "Folder4": "New Experiments/topk_output/five",
    "Folder5": "New Experiments/diversek_continual_output/five",
    "Folder6": "New Experiments/topk_continual_output",
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

# Colors and labels
colors = {
    "Folder2": "#ff7f0e",
    "Folder3": "#2ca02c",
    "Folder4": "#d62728",
    "Folder5": "#9467bd",
    "Folder6": "#8c564b"
}
linestyles = {
    "Folder2": "-",
    "Folder3": "-",
    "Folder4": "-",
    "Folder5": "-",
    "Folder6": "-"
}
labels = {
    "Folder2": "fair-topk",
    "Folder3": "topk-continual-static-lambda",
    "Folder4": "topk",
    "Folder5": "fair-topk-continual-lambda",
    "Folder6": "topk-continual"
}

# Y-axis limits for each metric
y_limits = {
    "t_rate": (0, 1.6),
    "model_shift": (0, 10),
    "acc": (0.6, 1)
}
# Compute adaptive y-limits for "model_shift"
all_model_shift_values = []
for folder_name in folder_paths.keys():
    for dataset in datasets:
        all_model_shift_values.extend(data_dict[folder_name][dataset]["model_shift"])

# Set adaptive limits
if all_model_shift_values:
    min_shift, max_shift = min(all_model_shift_values), max(all_model_shift_values)
    margin = (max_shift - min_shift) * 0.1  # 10% margin
    y_limits["model_shift"] = (max(0, min_shift - margin), max_shift + margin)
else:
    y_limits["model_shift"] = (0, 10)  # Default fallback in case no data exists

# Store legend handles for separate legend plot
legend_handles = []
legend_labels = []

# Generate separate figures for each metric
for metric in metrics_list:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), constrained_layout=True)

    for i, dataset in enumerate(datasets):
        ax = axes[i]

        for folder_name in folder_paths.keys():
            values = data_dict[folder_name][dataset][metric]
            if values:
                line, = ax.plot(values, linestyle=linestyles[folder_name], alpha=0.8,
                                color=colors[folder_name], linewidth=2, label=labels[folder_name])

                # Collect legend handles only once
                if labels[folder_name] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(labels[folder_name])

        # Set y-axis limit dynamically based on the metric
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])

        ax.grid(True, linestyle=":", alpha=0.5)
        if metric == "t_rate":
            ax.set_ylabel(f"Dataset: {dataset}", fontsize=16, fontweight="normal")

    # Add title
    if metric == "t_rate":
        fig.suptitle("Test Acceptance Rate", fontsize=16, fontweight="normal")
    elif metric == "model_shift":
        fig.suptitle("Model Shift", fontsize=16, fontweight="normal")
    elif metric == "acc":
        fig.suptitle("Short-Term Accuracy", fontsize=16, fontweight="normal")
    

    # Save the individual metric plot
    plt.savefig(f"Result/{metric}_comparison.png")
    plt.show()

# Create a separate legend plot
fig_legend, ax_legend = plt.subplots(figsize=(8, 2))
ax_legend.axis("off")  # Hide axes

# Create a legend
ax_legend.legend(legend_handles, legend_labels, loc="center", fontsize=12, frameon=True, ncol=2)

# Save the legend as a separate plot
plt.savefig("Result/legend_plot.png")
plt.show()
