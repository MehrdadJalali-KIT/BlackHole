import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths and configuration
folder_path = "BH_Datasets"
output_folder = "Metal_Distribution_Plots"
os.makedirs(output_folder, exist_ok=True)

thresholds = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']
file_prefix = "remaining_node_features_t"
file_suffix = "_r0.csv"
original_file = os.path.join(folder_path, f"{file_prefix}0.00{file_suffix}")

# Load original dataset
if not os.path.exists(original_file):
    print(f"Original file {original_file} not found.")
    exit()
original_df = pd.read_csv(original_file)

# Identify top 5 metals in the original dataset
top5_metals = original_df['metal'].value_counts().nlargest(5).index.tolist()

# Precompute original metal distribution (normalized)
original_counts = original_df['metal'].value_counts(normalize=True)
original_top5_dist = original_counts[top5_metals]

# Loop through thresholds and create plots
for thresh in thresholds:
    file_name = os.path.join(folder_path, f"{file_prefix}{thresh}{file_suffix}")
    if not os.path.exists(file_name):
        print(f"File {file_name} not found. Skipping threshold {thresh}.")
        continue

    thresh_df = pd.read_csv(file_name)
    label = thresh

    # Count frequencies of top 5 metals in the representative set
    rep_counts = thresh_df['metal'].value_counts(normalize=True)
    rep_top5_dist = rep_counts.reindex(top5_metals).fillna(0)

    # Combine into a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Metal': top5_metals,
        'Original': original_top5_dist.values,
        f'Threshold {label}': rep_top5_dist.values
    }).melt(id_vars='Metal', var_name='Dataset', value_name='Proportion')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Metal', y='Proportion', hue='Dataset',
                palette={'Original': '#4682B4', f'Threshold {label}': 'black'})

    #plt.title(f'Metal Distribution: Original vs. Threshold {label}', fontsize=14)
    plt.xlabel('Metal Atom', fontsize=20)
    plt.ylabel('Proportion of MOFs', fontsize=20)
    plt.ylim(0, 0.2)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.legend(title='', fontsize=14)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_folder, f"metal_dist_thresh_{label.replace('.', '')}.png")
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"Saved plot: {output_path}")

print("All metal distribution plots saved.")