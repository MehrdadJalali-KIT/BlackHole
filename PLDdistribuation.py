import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define the folder path and file pattern
folder_path = "BH_Datasets"
output_folder = "PLD_Plots"
os.makedirs(output_folder, exist_ok=True)

thresholds = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']
file_prefix = "remaining_node_features_t"
file_suffix = "_r0.csv"

# Load the original dataset (threshold 0.00)
original_file = os.path.join(folder_path, f"{file_prefix}0.00{file_suffix}")
if not os.path.exists(original_file):
    print(f"Error: Original file {original_file} not found. Exiting.")
    exit()

try:
    original_df = pd.read_csv(original_file)
    original_df['Threshold'] = 'Original'
except Exception as e:
    print(f"Error reading {original_file}: {e}. Exiting.")
    exit()

# Set white background and grid
plt.style.use('default')
sns.set_style("whitegrid")

# Loop through each threshold and create/save plot
for thresh in thresholds:
    file_name = os.path.join(folder_path, f"{file_prefix}{thresh}{file_suffix}")
    if not os.path.exists(file_name):
        print(f"Warning: File {file_name} not found. Skipping threshold {thresh}.")
        continue

    label = thresh  # e.g., "0.80"
    try:
        thresh_df = pd.read_csv(file_name)
        thresh_df['Threshold'] = label
    except Exception as e:
        print(f"Error reading {file_name}: {e}. Skipping threshold {thresh}.")
        continue

    # Combine with original data
    combined_df = pd.concat([
        original_df[['Pore Limiting Diameter', 'Threshold']],
        thresh_df[['Pore Limiting Diameter', 'Threshold']]
    ], ignore_index=True)

    # Custom color palette
    palette = {
        'Original': '#4682B4',  # SteelBlue
        label: 'black'
    }

    # Create plot
    plt.figure(figsize=(12, 7))
    sns.histplot(data=combined_df, x='Pore Limiting Diameter', hue='Threshold',
                 stat='probability', bins=30, alpha=0.8, edgecolor='black',
                 linewidth=0.5, palette=palette)

    # Customize
    plt.title(f'Original vs. Threshold {label}', fontsize=22, fontweight='bold')
    plt.xlabel('Pore Limiting Diameter (Å)', fontsize=20)
    plt.ylabel('Proportional Density', fontsize=20)
    plt.xlim(0, 20)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks([0.05, 0.10, 0.15, 0.20, 0.25], fontsize=18, fontweight='bold')  # Exclude 0 from y-ticks
    plt.legend().remove()  # Remove legend entirely

    # Save as high-resolution PNG
    output_path = os.path.join(output_folder, f"pld_histogram_thresh_{label.replace('.', '')}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, format='png')
    plt.close()  # Close to avoid overlapping plots

    print(f"Saved: {output_path}")

print("All plots saved successfully.")