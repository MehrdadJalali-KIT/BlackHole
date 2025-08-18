import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pubchempy import get_compounds
import json
from datetime import datetime

# Setup
folder_path = "BH_Datasets"
top_output_folder = "Linker_Distribution_Plots"
least_output_folder = "Least_Linker_Distribution_Plots"
os.makedirs(top_output_folder, exist_ok=True)
os.makedirs(least_output_folder, exist_ok=True)

thresholds = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90']
file_prefix = "remaining_node_features_t"
file_suffix = "_r0.csv"
original_file = os.path.join(folder_path, f"{file_prefix}0.00{file_suffix}")

# Load or create cache for SMILES to IUPAC conversion
cache_file = "smiles_iupac_cache.json"
smiles_name_map = {}
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        smiles_name_map = json.load(f)

def smiles_to_iupac(smiles):
    if smiles not in smiles_name_map:
        try:
            compounds = get_compounds(smiles, namespace='smiles')
            smiles_name_map[smiles] = compounds[0].iupac_name if compounds and compounds[0].iupac_name else smiles
            print(f"Converted {smiles} to {smiles_name_map[smiles]}")
        except Exception as e:
            smiles_name_map[smiles] = smiles
            print(f"Failed to convert {smiles}: {e}")
    with open(cache_file, 'w') as f:
        json.dump(smiles_name_map, f)
    return smiles_name_map[smiles]

# Load original dataset and validate
if not os.path.exists(original_file):
    print(f"Original file {original_file} not found at {datetime.now().strftime('%H:%M:%S %Z on %B %d, %Y')}")
    exit()
original_df = pd.read_csv(original_file)
if 'linker SMILES' not in original_df.columns:
    print("Column 'linker SMILES' not found in original dataset.")
    exit()

# Get top 5 and least 5 SMILES strings
top5_smiles = original_df['linker SMILES'].value_counts().nlargest(5).index.tolist()
least5_smiles = original_df['linker SMILES'].value_counts().nsmallest(5).index.tolist()
all_smiles = set(top5_smiles + least5_smiles)
print(f"Top 5 SMILES: {top5_smiles}")
print(f"Least 5 SMILES: {least5_smiles}")

# Build SMILES to IUPAC map
for smi in all_smiles:
    smiles_to_iupac(smi)

# Get original distribution
original_counts = original_df['linker SMILES'].value_counts(normalize=True)
original_top5_dist = original_counts[top5_smiles]
original_least5_dist = original_counts[least5_smiles]

# Loop through thresholds
for thresh in thresholds:
    file_path = os.path.join(folder_path, f"{file_prefix}{thresh}{file_suffix}")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found at threshold {thresh}. Skipping.")
        continue

    df = pd.read_csv(file_path)
    if 'linker SMILES' not in df.columns:
        print(f"Column 'linker SMILES' not found in {file_path}. Skipping.")
        continue
    label = thresh

    # --- Plot for Top 5 Linkers ---
    rep_counts = df['linker SMILES'].value_counts(normalize=True)
    rep_top5_dist = rep_counts.reindex(top5_smiles).fillna(0)

    plot_df_top = pd.DataFrame({
        'Linker': [smiles_name_map[smi] for smi in top5_smiles],
        'Original': original_top5_dist.values,
        f'Threshold {label}': rep_top5_dist.values
    }).melt(id_vars='Linker', var_name='Dataset', value_name='Proportion')
    print(f"Top 5 plot data for threshold {label}: {plot_df_top}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df_top, y='Linker', x='Proportion', hue='Dataset',
                palette={'Original': '#4682B4', f'Threshold {label}': 'black'})
    plt.ylabel('Linker (IUPAC Name)', fontsize=14, labelpad=20)
    plt.xlabel('Proportion of MOFs', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.2)
    plt.yticks(rotation=0)
    plt.legend(title='', loc='best')
    plt.tight_layout()

    output_path = os.path.join(top_output_folder, f"linker_dist_thresh_{label.replace('.', '')}.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top 5 linkers plot: {output_path}")

    # --- Plot for Least 5 Linkers ---
    rep_least5_dist = rep_counts.reindex(least5_smiles).fillna(0)

    plot_df_least = pd.DataFrame({
        'Linker': [smiles_name_map[smi] for smi in least5_smiles],
        'Original': original_least5_dist.values,
        f'Threshold {label}': rep_least5_dist.values
    }).melt(id_vars='Linker', var_name='Dataset', value_name='Proportion')
    print(f"Least 5 plot data for threshold {label}: {plot_df_least}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df_least, y='Linker', x='Proportion', hue='Dataset',
                palette={'Original': '#4682B4', f'Threshold {label}': 'black'})
    plt.ylabel('Linker (IUPAC Name)', fontsize=14, labelpad=20)
    plt.xlabel('Proportion of MOFs', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, max(plot_df_least['Proportion'].max(), 0.01) * 1.2)  # Dynamic limit with minimum 0.01
    plt.yticks(rotation=0)
    plt.legend(title='', loc='best')
    plt.tight_layout()

    output_path = os.path.join(least_output_folder, f"least_linker_dist_thresh_{label.replace('.', '')}.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved least 5 linkers plot: {output_path}")

print(f"All linker distribution plots saved at {datetime.now().strftime('%H:%M:%S %Z on %B %d, %Y')}.")