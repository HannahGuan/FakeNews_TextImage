"""
Check class distribution in the dataset to identify imbalance.

Author: Xinru Pan
Time: 2025-11-27
"""
import csv
from collections import Counter

def check_balance(csv_path, dataset_name):
    """Check class distribution for all label types."""
    print('='*70)
    print(f'{dataset_name} - Class Distribution')
    print('='*70)

    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f'Total samples: {len(rows)}')

    for label_type in ['2_way', '3_way', '6_way']:
        label_col = f'{label_type}_label'

        # Check if column exists
        if label_col not in rows[0]:
            continue

        print(f'\n{label_type.upper()} Classification:')
        print('-' * 50)

        # Count labels (excluding NaN/empty values)
        labels = [row[label_col] for row in rows if row[label_col] and row[label_col] != 'nan']
        counts = Counter(labels)
        total = len(labels)

        # Print counts and percentages
        for label in sorted(counts.keys()):
            count = counts[label]
            percentage = (count / total) * 100
            print(f'  Class {label}: {count:6d} samples ({percentage:5.2f}%)')

        # Check if imbalanced
        percentages = [(count / total) * 100 for count in counts.values()]
        min_percentage = min(percentages)
        max_percentage = max(percentages)

        print(f'\n  Minority class has {min_percentage:.2f}%')
      
    print()

if __name__ == '__main__':
    # Check training set
    check_balance('/Users/panxinru/Desktop/Stanford/2025 Fall/CS230/Group Project/Data/train_sampled_with_images.csv', 'TRAINING SET')
    
    check_balance('/Users/panxinru/Desktop/Stanford/2025 Fall/CS230/Group Project/Data/dev_sampled_with_images.csv', 'DEV SET')