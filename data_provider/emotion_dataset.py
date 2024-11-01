import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    """Custom Dataset for loading the audiovisual features"""
    def __init__(self, features, labels, verbose=False):
        self.features = torch.FloatTensor(features)

        # Create label encoder
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Convert string labels to indices
        encoded_labels = [self.label_to_idx[label] for label in labels]
        self.labels = torch.LongTensor(encoded_labels)

        # Store number of classes
        self.num_classes = len(unique_labels)

        if verbose:
            print('Dataset shape:', self.features.shape)

            # Print the label mapping
            print("Label Encoding Mapping:")
            print("-" * 20)
            for label, idx in self.label_to_idx.items():
                print(f"'{label}' -> {idx}")
            print("-" * 20)

            print(f"Total number of classes: {self.num_classes}\n")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def decode_labels(self, encoded_labels):
        """Convert numeric labels back to original string labels"""
        if isinstance(encoded_labels, torch.Tensor):
            encoded_labels = encoded_labels.cpu().numpy()
        return [self.idx_to_label[idx] for idx in encoded_labels]