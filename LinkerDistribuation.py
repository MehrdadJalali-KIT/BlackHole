import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pubchempy import get_compounds
from collections import defaultdict

# Setup
folder_path = "BH_Datasets"
output_folder = "Linker_Distribution_Plots"
os.makedirs(output_folder, exist_ok=True)

thresholds = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']
file_prefix = "remaining_node_features_t"
file_suffix = "_r0.csv"
original_file = os.path.join(folder_path, f"{file_prefix}0.00{file_suffix}")

# Load original dataset
original_df = pd.read_csv(original_file)

# Get top 5 most common SMILES strings
top5_smiles = original_df['linker SMILES'].value_counts().nlargest(5).index.tolist()

# Convert SMILES → IUPAC name via PubChem with full names
def smiles_to_iupac(smiles):
    try:
        compounds = get_compounds(smiles, namespace='smiles')
        return compounds[0].iupac_name if compounds and compounds[0].iupac_name else "Unrecognized"
    except:
        return "Unrecognized"

# Build map once
smiles_name_map = {smi: smiles_to_iupac(smi) for smi in top5_smiles}

# Get original distribution
original_counts = original_df['linker SMILES'].value_counts(normalize=True)
original_top5_dist = original_counts[top5_smiles]

# Loop through thresholds
for thresh in thresholds:
    file_path = os.path.join(folder_path, f"{file_prefix}{thresh}{file_suffix}")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    label = thresh

    # Get representative distribution
    rep_counts = df['linker SMILES'].value_counts(normalize=True)
    rep_top5_dist = rep_counts.reindex(top5_smiles).fillna(0)

    # Prepare data for plotting
    plot_df = pd.DataFrame({
        'Linker': [smiles_name_map[smi] for smi in top5_smiles],
        'Original': original_top5_dist.values,
        f'Threshold {label}': rep_top5_dist.values
    }).melt(id_vars='Linker', var_name='Dataset', value_name='Proportion')

    # Plot
    plt.figure(figsize=(14, 6))  # Increased width to accommodate longer names
    ax = sns.barplot(data=plot_df, y='Linker', x='Proportion', hue='Dataset',
                     palette={'Original': '#4682B4', f'Threshold {label}': 'black'})
    plt.xlabel('Proportion of MOFs', fontsize=20)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    ax.set_ylabel('')  # Remove the "Linker" y-axis label
    plt.gca().set_xticklabels([f'{x:.2f}' for x in plt.gca().get_xticks()])  # Format x-axis to two decimal places
    plt.legend(title='', loc='lower right', fontsize=14)  # Add legend at bottom right
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_folder, f"linker_dist_thresh_{label.replace('.', '')}.png")
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"Saved: {output_path}")