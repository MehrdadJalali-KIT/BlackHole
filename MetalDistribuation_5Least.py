import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths and configuration
folder_path = "BH_Datasets"
top_output_folder = "Metal_Distribution_Plots"
least_output_folder = "Least_Metal_Distribution_Plots"
os.makedirs(top_output_folder, exist_ok=True)
os.makedirs(least_output_folder, exist_ok=True)

thresholds = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']
file_prefix = "remaining_node_features_t"
file_suffix = "_r0.csv"
original_file = os.path.join(folder_path, f"{file_prefix}0.00{file_suffix}")

# Load original dataset
if not os.path.exists(original_file):
    print(f"Original file {original_file} not found.")
    exit()
original_df = pd.read_csv(original_file)

# Identify top 5 and least 5 metals in the original dataset
top5_metals = original_df['metal'].value_counts().nlargest(5).index.tolist()
least5_metals = original_df['metal'].value_counts().nsmallest(5).index.tolist()
print(f"Top 5 metals: {top5_metals}")
print(f"Least 5 metals: {least5_metals}")

# Precompute original metal distribution (normalized)
original_counts = original_df['metal'].value_counts(normalize=True)
original_top5_dist = original_counts[top5_metals]
original_least5_dist = original_counts[least5_metals]

# Loop through thresholds and create plots for top 5 metals
for thresh in thresholds:
    file_name = os.path.join(folder_path, f"{file_prefix}{thresh}{file_suffix}")
    if not os.path.exists(file_name):
        print(f"File {file_name} not found. Skipping threshold {thresh}.")
        continue

    thresh_df = pd.read_csv(file_name)
    label = thresh

    # --- Plot for Top 5 Metals ---
    # Count frequencies of top 5 metals in the representative set
    rep_counts = thresh_df['metal'].value_counts(normalize=True)
    rep_top5_dist = rep_counts.reindex(top5_metals).fillna(0)

    # Combine into a DataFrame for plotting
    plot_df_top = pd.DataFrame({
        'Metal': top5_metals,
        'Original': original_top5_dist.values,
        f'Threshold {label}': rep_top5_dist.values
    }).melt(id_vars='Metal', var_name='Dataset', value_name='Proportion')

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df_top, x='Metal', y='Proportion', hue='Dataset',
                palette={'Original': '#4682B4', f'Threshold {label}': 'black'})
    plt.xlabel('Metal Atom', fontsize=16)
    plt.ylabel('Proportion of MOFs', fontsize=16)
    plt.ylim(0, 0.2)  # Set y-axis to 0.2 to focus on typical metal proportions
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='')
    plt.tight_layout()

    # Save
    output_path = os.path.join(top_output_folder, f"metal_dist_thresh_{label.replace('.', '')}.jpg")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved top 5 metals plot: {output_path}")

    # --- Plot for Least 5 Metals ---
    # Count frequencies of least 5 metals in the representative set
    rep_least5_dist = rep_counts.reindex(least5_metals).fillna(0)

    # Combine into a DataFrame for plotting
    plot_df_least = pd.DataFrame({
        'Metal': least5_metals,
        'Original': original_least5_dist.values,
        f'Threshold {label}': rep_least5_dist.values
    }).melt(id_vars='Metal', var_name='Dataset', value_name='Proportion')

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df_least, x='Metal', y='Proportion', hue='Dataset',
                palette={'Original': '#4682B4', f'Threshold {label}': 'black'})
    plt.xlabel('Metal Atom', fontsize=16)
    plt.ylabel('Proportion of MOFs', fontsize=16)
    plt.ylim(0, max(plot_df_least['Proportion']) * 1.2)  # Dynamic y-axis for rare metals
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title='')
    plt.tight_layout()

    # Save
    output_path = os.path.join(least_output_folder, f"least_metal_dist_thresh_{label.replace('.', '')}.jpg")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved least 5 metals plot: {output_path}")

print("All metal distribution plots saved.")