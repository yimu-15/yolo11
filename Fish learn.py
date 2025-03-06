from IPython.display import HTML
import base64
import os
HOME = os.getcwd()
print(HOME)

# Path to the dataset
dataset_path = '/kaggle/input/fish-detection'

# Initialize a dictionary to count the number of files in each subset
subset_counts = {'train': 0, 'test': 0, 'valid': 0}

# Walk through train, test, and valid directories to count files
for subset in ['train', 'test', 'valid']:
    subset_path = os.path.join(dataset_path, subset)
    for dirname, _, filenames in os.walk(subset_path):
        for filename in filenames:
            if filename.endswith('.txt'):  # Only count .txt files
                subset_counts[subset] += 1

# Print the number of images in each subset (train, test, valid)
print(f"Training set size: {subset_counts['train']}")
print(f"Test set size: {subset_counts['test']}")
print(f"Validation set size: {subset_counts['valid']}")
