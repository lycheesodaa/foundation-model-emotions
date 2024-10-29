from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm


class TimeseriesDataset(Dataset):
    """Custom Dataset for loading the audiovisual features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        print(self.features.shape)

        # Create label encoder
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Print the label mapping
        print("\nLabel Encoding Mapping:")
        print("-" * 20)
        for label, idx in self.label_to_idx.items():
            print(f"'{label}' -> {idx}")
        print("-" * 20)

        # Convert string labels to indices
        encoded_labels = [self.label_to_idx[label] for label in labels]
        self.labels = torch.LongTensor(encoded_labels)

        # Store number of classes
        self.num_classes = len(unique_labels)
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm1 = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(
            hidden_dim * self.num_directions, 
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * self.num_directions, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out[:, -1, :])  # Take only the last output
        fc1_out = self.relu(self.fc1(lstm2_out))
        fc1_out = self.dropout3(fc1_out)
        out = self.fc2(fc1_out)
        return out

class FCDNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(FCDNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def prepare_data(directory, feature_type="dynamic"):
    print(f'Preparing data from {directory}...')

    features = {}
    labels = []
    speakers = []
    for speaker in range(1, 6):
        for gender in ['F', 'M']:
            speaker_id = f"{speaker}{gender}"

            filepath = Path(directory) / feature_type / f"audio_visual_speaker_{speaker_id}.csv"
            feature_set = pd.read_pickle(filepath)
            features[speaker_id] = np.vstack([
                feature_set["features"][idx] for idx in range(feature_set.shape[0])
            ])
            if np.isnan(features[speaker_id]).any():
                print(f"Warning: Null values found in features for speaker {speaker_id}")
                features = np.nan_to_num(features, nan=0.0)
            labels.extend([
                feature_set["UF_label"][idx] for idx in range(feature_set.shape[0])
            ])
            speakers.extend([speaker_id] * feature_set.shape[0])

    full_feature_set = np.vstack(list(features.values()))

    data_instances = len(labels)
    sequence_length = int(full_feature_set.shape[0] / data_instances)
    feature_dimension = 895

    if feature_type == "dynamic":
        full_feature_set = full_feature_set.reshape(
            data_instances, sequence_length, feature_dimension
        )

    return full_feature_set, np.asarray(labels), np.asarray(speakers)


class RunDeepLearning:
    """
    This class will run the deep learning algorithms, LSTM/BiLSTM/FC-DNN
    from the speaker-wise processed data
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.max_grad_norm = 1.0  # For gradient clipping

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for features, labels in tqdm(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return running_loss / len(val_loader), correct / total

    def train_model(self, model, train_loader, val_loader, num_epochs=50, patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            print(f'Epoch {epoch} - Validation Loss: {val_loss} | Validation Accuracy: {val_acc}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Validation has not improved in {patience} runs. Early stopping...')
                print(f'Best validation loss: {val_loss}')
                break
                
        return model

    def run_model(self, features, labels, speaker_id, model_type="lstm", bidirectional=False):
        logo = LeaveOneGroupOut()
        predict_probability = {}
        test_GT = {}
        model_pred = {}
        confusion_matrices = {}
        unweighted_recall = {}
        
        input_dim = features.shape[2] if model_type == "lstm" else features.shape[1]

        # leave-one-out cross validation
        speaker = 0
        for train_idx, test_idx in logo.split(features, labels, speaker_id):
            # Create the dataset
            dataset = TimeseriesDataset(features, labels)

            if model_type == "lstm":
                model = LSTMModel(
                    input_dim=input_dim,
                    num_classes=dataset.num_classes,
                    bidirectional=bidirectional
                ).to(self.device)
            else:
                model = FCDNNModel(
                    input_dim=input_dim,
                    num_classes=dataset.num_classes
                ).to(self.device)

            # Create train/val split from training data
            train_size = int(0.8 * len(train_idx))
            train_subset = train_idx[:train_size]
            val_subset = train_idx[train_size:]

            # Create samplers for each set
            train_sampler = SubsetRandomSampler(train_subset)
            val_sampler = SubsetRandomSampler(val_subset)
            test_sampler = SubsetRandomSampler(test_idx)

            # Create data loaders
            train_loader = DataLoader(
                dataset, batch_size=128, sampler=train_sampler
            )
            val_loader = DataLoader(
                dataset, batch_size=128, sampler=val_sampler
            )
            test_loader = DataLoader(
                dataset, batch_size=128, sampler=test_sampler
            )
            
            # Train the model
            model = self.train_model(model, train_loader, val_loader)
            
            # Evaluate on test set
            model.eval()
            test_predictions = []
            test_labels = []
            
            with torch.no_grad():
                for features_batch, labels_batch in test_loader:
                    features_batch = features_batch.to(self.device)
                    outputs = model(features_batch)
                    probabilities = torch.softmax(outputs, dim=1)
                    test_predictions.extend(probabilities.cpu().numpy())
                    test_labels.extend(labels_batch.numpy())
            
            test_predictions = np.array(test_predictions)
            test_labels = np.array(test_labels)

            # evaluation metrics
            predict_probability[speaker] = test_predictions
            test_GT[speaker] = test_labels
            model_pred[speaker] = np.argmax(test_predictions, axis=1)
            
            confusion_matrices[speaker] = confusion_matrix(
                test_labels, model_pred[speaker]
            )
            unweighted_recall[speaker] = recall_score(
                test_labels, model_pred[speaker], average='macro'
            )
            
            speaker += 1
            
        return test_GT, model_pred, predict_probability, confusion_matrices, unweighted_recall


features, label, speaker_group = prepare_data(directory='Files/UF_His_data/step_1')
forecast = RunDeepLearning()
forecast.run_model(features, label, speaker_group)
