import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Normalize temporal and nasal distance columns
# on dataframe
# Parameters:
#   - df: pandas dataframe
def normalizeDistances(df):
    df['temporal_distance_normalized'] = df['temporal_distance'] / df['disc_diameter']
    df['nasal_distance_normalized'] = df['nasal_distance'] / df['disc_diameter']

    return df

# Plot two histograms for nasal/temporal distance
# measures occurrencies on BRSet
# Parameters:
#   - df: filtered (by confidence) dataframe with interest statistics
#   - structure: fovea/optic disc
#   - distance: temporal/nasal
#   - save_path: base path to save the histograms
def plotHistogramsTwoPlots(df, structure, distance, save_path):
    adequate_nasal = df[df['quality_label'] == 'Adequate'][f'{distance}_distance_normalized']
    inadequate_nasal = df[df['quality_label'] == 'Inadequate'][f'{distance}_distance_normalized']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # adequate images histogram
    axes[0].hist(adequate_nasal, bins=40, color='green', alpha=0.6)
    axes[0].set_title(f'Adequate by Image Field: {structure.capitalize()}-{distance.capitalize()} Edge Distance')
    axes[0].set_xlabel(f'Normalized {structure.capitalize()}-{distance.capitalize()} Edge Distance')
    axes[0].set_ylabel('Density')

    # inadequate images histogram
    axes[1].hist(inadequate_nasal, bins=40, color='red', alpha=0.6)
    axes[1].set_title(f'Inadequate by Image Field: {structure.capitalize()}-{distance.capitalize()} Edge Distance')
    axes[1].set_xlabel(f'Normalized {structure.capitalize()}-{distance.capitalize()} Edge Distance')

    plt.tight_layout()
    plt.savefig(f"{save_path}/{distance}HistogramTwoPlots.png")

# Plot one histogram for nasal/temporal distance
# measures occurrencies on BRSet
# Parameters:
#   - df: filtered (by confidence) dataframe with interest statistics
#   - structure: fovea/optic disc
#   - distance: temporal/nasal
#   - save_path: base path to save the histograms
def plotHistogramsOnePlot(df, structure, distance, save_path):
    adequate_nasal = df[df['quality_label'] == 'Adequate'][f'{distance}_distance_normalized']
    inadequate_nasal = df[df['quality_label'] == 'Inadequate'][f'{distance}_distance_normalized']

    plt.figure(figsize=(8, 5))

    plt.hist(adequate_nasal, bins=40, color='green', density=True, alpha=0.25, label='Adequate images by Image Field')
    plt.hist(inadequate_nasal, bins=40, color='red', density=True, alpha=0.25, label='Inadequate images by Image Field')
    
    sns.kdeplot(adequate_nasal, color='green', linestyle='-')
    sns.kdeplot(inadequate_nasal, color='red', linestyle='-')

    plt.title(f'{structure.capitalize()}-{distance.capitalize()} Edge Distance Histogram')
    plt.xlabel(f'Normalized {structure.capitalize()}-{distance.capitalize()} Edge Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{distance}HistogramOnePlot.png")

def main(args):
    df = pd.read_csv(args.data)
    df = normalizeDistances(df)

    countAdequate = df[df['quality_label'] == 'Adequate'].shape[0]
    countInadequate = df[df['quality_label'] == 'Inadequate'].shape[0]

    # images our model was not able to detect fovea or optic disc
    missingImages = args.dataset_size - (countAdequate + countInadequate)

    filtered_df = df[(df['od_confidence'] > 0.8) & (df['fovea_confidence'] > 0.75)]

    countAdequate = filtered_df[filtered_df['quality_label'] == 'Adequate'].shape[0]
    countInadequate = filtered_df[filtered_df['quality_label'] == 'Inadequate'].shape[0]

    print(f"Dados antes de filtrar por confiança: {countAdequate} adequadas, {countInadequate} inadequadas")
    print(f"Dados depois de filtrar por confiança: {countAdequate} adequadas, {countInadequate} inadequadas")

    plotHistogramsOnePlot(filtered_df, structure='optic disc', distance='nasal', save_path=args.save_path)
    plotHistogramsTwoPlots(filtered_df, structure='optic disc', distance='nasal', save_path=args.save_path)

    plotHistogramsOnePlot(filtered_df, structure='fovea', distance='temporal', save_path=args.save_path)
    plotHistogramsTwoPlots(filtered_df, structure='fovea', distance='temporal', save_path=args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset analysis for BRSet")

    parser.add_argument('--dataset-size', type=int, default=16266, help='Total images on the dataset')
    parser.add_argument('--data', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/data/retinalInformation.csv', help='Data path for statistical analysis')
    parser.add_argument('--save-path', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/data/images', help='Where to save generated graphics')

    args = parser.parse_args()

    main(args)