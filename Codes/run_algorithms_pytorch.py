import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

class TimeseriesDataset(Dataset):
    """Custom Dataset for loading the audio-visual features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

class RunDeepLearning:
    """
    This class will run the deep learning algorithms, LSTM and BLSTM
    from the speakerwise processed data
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def prepare_data(self, directory, feature_type="dynamic"):
        Features = {}
        Labels = []
        Speakers = []
        for speaker in range(1, 11):
            filepath = os.path.join(directory, f"audio_visual_speaker_{speaker}.csv")
            feature_set = pd.read_pickle(filepath)
            Features[speaker] = np.vstack([
                feature_set["features"][idx] for idx in range(feature_set.shape[0])
            ])
            Labels.extend([
                feature_set["UF_label"][idx] for idx in range(feature_set.shape[0])
            ])
            Speakers.extend([speaker] * feature_set.shape[0])

        Full_feature_set = np.vstack([Features[speaker] for speaker in range(1, 11)])

        data_instances = len(Labels)
        sequence_length = int(Full_feature_set.shape[0] / data_instances)
        feature_dimension = 895
        
        if feature_type == "dynamic":
            Full_feature_set = Full_feature_set.reshape(
                data_instances, sequence_length, feature_dimension
            )
        
        return Full_feature_set, np.asarray(Labels), np.asarray(Speakers)

    def train_epoch(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
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
        
        speaker = 0
        for train_idx, test_idx in logo.split(features, labels, speaker_id):
            if model_type == "lstm":
                model = LSTMModel(
                    input_dim=input_dim,
                    bidirectional=bidirectional
                ).to(self.device)
            else:
                model = FCDNNModel(input_dim=input_dim).to(self.device)

            # Create dataset and dataloaders
            dataset = TimeseriesDataset(features, labels)
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            
            train_loader = DataLoader(
                dataset, batch_size=128, sampler=train_sampler
            )
            test_loader = DataLoader(
                dataset, batch_size=128, sampler=test_sampler
            )
            
            # Split train into train and validation
            train_size = int(0.8 * len(train_idx))
            val_size = len(train_idx) - train_size
            train_subset, val_subset = torch.utils.data.random_split(
                train_idx, [train_size, val_size]
            )
            
            val_loader = DataLoader(
                dataset, batch_size=128, sampler=SubsetRandomSampler(val_subset)
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