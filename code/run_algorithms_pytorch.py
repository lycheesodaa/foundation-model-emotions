import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from utils import confusion_matrix_plot, prepare_data
from emotion_dataset import EmotionDataset


class MaskedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(MaskedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x, mask=None):
        if mask is not None:
            # Create a packed sequence to handle masking
            lengths = mask.sum(dim=1).cpu()
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed_sequence)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, padding_value=0.0
            )
        else:
            output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm1 = MaskedLSTM(
            input_dim,
            hidden_dim,
            bidirectional=bidirectional
        )
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = MaskedLSTM(
            hidden_dim * self.num_directions,
            hidden_dim,
            bidirectional=bidirectional
        )
        self.dropout2 = nn.Dropout(0.5)

        lstm_output_dim = hidden_dim * self.num_directions
        self.fc1 = nn.Linear(lstm_output_dim, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        # Apply masking if provided (zeros in the mask indicate padded values)
        if mask is None:
            mask = (x.sum(dim=2) != 0)  # Create mask based on non-zero values

        lstm1_out, _ = self.lstm1(x, mask)
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, (hidden, _) = self.lstm2(lstm1_out, mask)

        # If bidirectional, concatenate the last hidden states from both directions
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]

        # Apply dropouts and fully connected layers
        fc1_out = self.relu(self.fc1(last_hidden))
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
            # print(features.shape)
            # print(labels.shape)

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

    def run_model(self, features, labels, speaker_ids, model_type="lstm", bidirectional=False):
        logo = LeaveOneGroupOut()
        predict_probability = {}
        test_GT = {}
        model_pred = {}
        confusion_matrices = {}
        unweighted_recall = {}
        
        input_dim = features.shape[2] if model_type == "lstm" else features.shape[1]

        # leave-one-out cross validation
        speaker = 0
        for train_idx, test_idx in logo.split(features, labels, speaker_ids):
            print(f'\n******** Cross-validating speaker {speaker}... ********')

            # Create the dataset
            dataset = EmotionDataset(features, labels)

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


if __name__ == "__main__":
    features, labels, speaker_group = prepare_data(directory='Files/UF_His_data/step_1')
    forecast = RunDeepLearning()
    test_gt, pred, pred_prob, confusion_matrices, uw_recall = forecast.run_model(
        features, labels, speaker_group, model_type='lstm', bidirectional=False
    )

    all_recall = []
    for i, speaker in enumerate(list(dict.fromkeys(speaker_group))):
        confusion_matrix_plot(
            str(speaker), test_gt[i], pred[i],
            ['Anger', 'Happy', 'Neutral', 'Sad'], normalize=False
        )
        all_recall.append(recall_score(test_gt[i], pred[i], average='macro'))

    print('Average UWR:', np.mean(all_recall * 100))
