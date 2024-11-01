import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from models.dnn import LSTMModel, FCDNNModel
from utils.utils import confusion_matrix_plot, prepare_data
from data_provider.emotion_dataset import EmotionDataset


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
