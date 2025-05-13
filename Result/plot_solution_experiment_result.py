import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the dataset and metrics
dataset = "credit"
metrics_list = ["avg_score", "model_shift", "acc", "failToRecourse", "avgRecourseCost", "t_rate", "avg_score_on_last_train", "recourse_cost_ratio"]

# Folder paths
folder_paths = {
    "Folder2": "New Experiments/diversek_output/5-15",
    "Folder3": "New Experiments/topk_continual_static_output/5-15",
    "Folder4": "New Experiments/topk_output/5-15",
    "Folder5": "New Experiments/diversek_continual_output/5-15",
    "Folder6": "New Experiments/topk_continual_output/5-15",
}

# folder_paths = {
#     "Folder2": "New Experiments/diversek_MLP_output/5-15",
#     "Folder3": "New Experiments/topk_continual_static_MLP_output/5-15",
#     "Folder4": "New Experiments/topk_MLP_output/5-15",
#     "Folder5": "New Experiments/diversek_continual_MLP_output/5-15",
#     "Folder6": "New Experiments/topk_continual_MLP_output/5-15",
# }

# Dictionary to store extracted data
data_dict = {
    folder: {metric: [] for metric in metrics_list}
    for folder in folder_paths
}

# Load data
for folder_name, folder_path in folder_paths.items():
    if not os.path.exists(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            parts = filename.split("_")
            if len(parts) < 6:
                continue

            dataset_name = parts[4]
            if dataset_name != dataset:
                continue

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            for metric in metrics_list:
                if metric == "recourse_cost_ratio":
                    if "avgNewRecourseCost" in df.columns and "avgOriginalRecourseCost" in df.columns:
                        try:
                            new = df["avgNewRecourseCost"].astype(float)
                            original = df["avgOriginalRecourseCost"].astype(float)
                            ratio = new / original.replace({0: np.nan})  # Avoid division by 0
                            ratio_cleaned = ratio.replace([np.inf, -np.inf], np.nan).dropna().tolist()
                            data_dict[folder_name][metric].extend(ratio_cleaned)
                        except Exception as e:
                            print(f"Error processing ratio in {file_path}: {e}")
                    continue

                # Continue with other metrics as before
                if metric in df.columns:
                    values = df[metric].astype(str).values
                    cleaned_values = []
                    skip_count = 1 if metric in ["acc", "model_shift", "failToRecourse", "avgRecourseCost"] else 0
                    values = values[skip_count:]

                    for val in values:
                        if "tensor" in val:
                            try:
                                num = float(val.replace("tensor(", "").replace(")", ""))
                                cleaned_values.append(num)
                            except ValueError:
                                print(f"Skipping invalid tensor value: {val}")
                        else:
                            try:
                                cleaned_values.append(float(val))
                            except ValueError:
                                print(f"Skipping invalid value: {val}")

                    data_dict[folder_name][metric].extend(cleaned_values)

# Ensure data is loaded correctly
print("Data dictionary preview:", data_dict)

# Colors and labels
colors = {
    "Folder2": "#ff7f0e",
    "Folder3": "#1f77b4",
    "Folder4": "#9467bd",
    "Folder5": "#d62728",
    "Folder6": "#2ca02c"
}
linestyles = {
    "Folder2": "-",
    "Folder3": "-",
    "Folder4": "-",
    "Folder5": "-",
    "Folder6": "-",
}
labels = {
    "Folder2": "Fair Top-k",
    "Folder3": "Top-k with static continual learning",
    "Folder4": "Top-k",
    "Folder5": "Fair Top-k with DCL",
    "Folder6": "Top-k with DCL",
}

# Y-axis limits for each metric
y_limits = {
    "avg_score": (5, -15),
    "model_shift": (0, 5),
    "acc": (0.5, 1)
}

# Compute adaptive y-limits for "model_shift"
all_model_shift_values = []
for folder_name in folder_paths.keys():
    all_model_shift_values.extend(data_dict[folder_name]["model_shift"])

# Set adaptive limits
if all_model_shift_values:
    min_shift, max_shift = min(all_model_shift_values), max(all_model_shift_values)
    margin = (max_shift - min_shift) * 0.1  # 10% margin
    y_limits["model_shift"] = (max(0, min_shift - margin), max_shift + margin)
else:
    y_limits["model_shift"] = (0, 0)  # Default fallback in case no data exists

# Store legend handles for separate legend plot
legend_handles = []
legend_labels = []

# Generate separate figures for each metric
for metric in metrics_list:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for folder_name in folder_paths.keys():
        values = data_dict[folder_name][metric]  # Access the correct metric directly
        if values:
            x_values = np.arange(1, len(values) + 1) if metric == "acc" or metric == "model_shift" else np.arange(len(values))
            line_alpha = 1.0 if folder_name in ["Folder2", "Folder5", "Folder6"] else 0.5
            line, = ax.plot(x_values, values, linestyle=linestyles[folder_name], alpha=line_alpha,
                        color=colors[folder_name], linewidth=2, label=labels[folder_name])

            # Collect legend handles only once
            if labels[folder_name] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(labels[folder_name])

    # Set y-axis limit dynamically based on the metric
    if metric == "acc":
        ax.margins(y=0.1)  # Adds 10% extra space above the highest point
    elif metric in y_limits:
        ax.set_ylim(y_limits[metric])

    ax.margins(y=0.1)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Only show side titles for "t_rate"
    if metric == "avg_score":
        
        ax.set_ylabel("Logit Sum", fontsize=18, fontweight="normal")
        ax.text(-0.25, 0.5, "Logistic Model", transform=ax.transAxes, 
                fontsize=23, rotation='vertical', va='center')


    # Add title
    title_map = {
    "avg_score": "Higher Standard",
    "model_shift": "Model Shift",
    "acc": "Short-Term Accuracy",
    "failToRecourse": "Recourse Failure Rate",
    "avgRecourseCost": "Average Recourse Cost",
    "t_rate": "Test Acceptance Rate",
    "avg_score_on_last_train": "Higher Standard",
    "recourse_cost_ratio": "Ratio of Effort"
    }


    ax.set_title(title_map.get(metric, ""), fontsize=23, fontweight="normal", pad=15)
    # fig.suptitle(title_map.get(metric, ""), fontsize=23, fontweight="normal")

    # Save the individual metric plot
    plt.savefig(f"Result/5-15L{metric}_comparison.png")
    plt.show()

# Create a separate legend plot
fig_legend, ax_legend = plt.subplots(figsize=(20,1.5))
ax_legend.axis("off")  # Hide axes

# Create a legend
preferred_order = ["Fair Top-k", "Fair Top-k with DCL", "Top-k with DCL"]

# Sort handles and labels based on the preferred order
sorted_legend = sorted(zip(legend_labels, legend_handles), key=lambda x: preferred_order.index(x[0]) if x[0] in preferred_order else len(preferred_order))

# Unzip the sorted legend
legend_labels, legend_handles = zip(*sorted_legend)

# Create the legend with the new order
ax_legend.legend(legend_handles, legend_labels, loc="upper center", 
                 fontsize=20, frameon=True, ncol=3)


# Save the legend as a separate plot
plt.savefig("Result/LTlegend_plot.png")
plt.show()
