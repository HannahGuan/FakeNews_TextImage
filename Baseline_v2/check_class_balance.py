"""
Check class distribution in the dataset to identify imbalance.

Author: Xinru Pan
Time: 2025-11-27
"""
import pandas as pd

def check_balance(csv_path, dataset_name):
    """Check class distribution for all label types."""

    # Read CSV
    df = pd.read_csv(csv_path)

    print(f'Total samples: {len(df)}')

    for label_type in ['2_way', '3_way', '6_way']:
        label_col = f'{label_type}_label'

        # count labels
        series = df[label_col].dropna()
        counts = series.value_counts()
        percentages = series.value_counts(normalize=True) * 100

        print(f'\n{label_type} labels:')

        # Print counts and percentages for each class
        for label in counts.index:
            count = counts[label]
            percentage = percentages[label]
            print(f'Class {label}:{count} samples({percentage:.2f}%)')

    print()

if __name__ == '__main__':
    # Check training set
    check_balance('/Users/panxinru/Desktop/Stanford/2025 Fall/CS230/Group Project/Data/train_sampled_with_images.csv', 'TRAINING SET')
    check_balance('/Users/panxinru/Desktop/Stanford/2025 Fall/CS230/Group Project/Data/dev_sampled_with_images.csv', 'DEV SET')
