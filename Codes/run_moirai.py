import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score, confusion_matrix
from uni2ts.model.moirai import MoiraiModule

from utils import prepare_data, confusion_matrix_plot
from emotion_predictor import EmotionForecastModel, EmotionPredictor
from run_algorithms_pytorch import TimeseriesDataset


def train_emotion_forecast_model(features, labels, speaker_ids, prediction_length=24, num_epochs=50, patience=10):
    # Model parameters
    target_dim = features.shape[2]  # Feature dimension
    context_length = features.shape[1]  # Sequence length
    num_emotions = len(set(labels))

    # Initialize model
    model = EmotionForecastModel(
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,  # Adjust if you have dynamic features
        past_feat_dynamic_real_dim=0,  # Adjust if you have past dynamic features
        context_length=context_length,
        num_emotions=num_emotions,
        hidden_dim=256,
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-large"),
    )

    # Setup cross-validation
    logo = LeaveOneGroupOut()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # Leave-one-speaker-out cross validation
    for fold, (train_idx, test_idx) in enumerate(logo.split(features, labels, speaker_ids)):
        print(f"\nTraining fold {fold} (Speaker {speaker_ids[test_idx[0]]})")

        # Create train/val split
        train_size = int(0.8 * len(train_idx))
        train_subset = train_idx[:train_size]
        val_subset = train_idx[train_size:]

        # Create data loaders
        train_loader = DataLoader(
            TimeseriesDataset(features[train_subset], labels[train_subset]),
            batch_size=128,
            shuffle=True
        )
        val_loader = DataLoader(
            TimeseriesDataset(features[val_subset], labels[val_subset]),
            batch_size=128
        )
        test_loader = DataLoader(
            TimeseriesDataset(features[test_idx], labels[test_idx]),
            batch_size=128
        )

        # Initialize training components
        predictor = EmotionPredictor(model, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = predictor.train_epoch(train_loader, optimizer, criterion)

            # Validate
            val_loss, val_uwr = predictor.validate(val_loader, criterion)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val UWR = {val_uwr:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt'))
        predictor = EmotionPredictor(model, device=device)

        # Get predictions
        all_forecasts = []
        all_probs = []
        all_labels = []

        for batch in test_loader:
            features_batch, labels_batch = batch
            forecasts, probs = predictor.predict(features_batch)
            all_forecasts.append(forecasts)
            all_probs.append(probs)
            all_labels.extend(labels_batch.numpy())

        # Store results
        results[speaker_ids[test_idx[0]]] = {
            'forecasts': torch.cat(all_forecasts, dim=0).numpy(),
            'probabilities': torch.cat(all_probs, dim=0).numpy(),
            'true_labels': all_labels
        }

    return results


# Example usage
if __name__ == "__main__":
    pred_len = 24

    # Prepare your data
    features, labels, speaker_ids = prepare_data(directory='Files/UF_His_data/step_1')

    # Train model and get results
    results = train_emotion_forecast_model(features, labels, speaker_ids, prediction_length=pred_len)

    all_recall = []
    # Print results for each speaker
    for speaker, result in results.items():
        predictions = np.argmax(result['probabilities'], axis=1)
        accuracy = np.mean(predictions == result['true_labels'])
        print(f"\nResults for Speaker {speaker}:")
        print(f"Accuracy: {accuracy:.4f}")

        # Confusion matrix
        # print("Confusion Matrix:")
        # print(confusion_matrix(result['true_labels'], predictions))
        confusion_matrix_plot(
            str(speaker), result['true_labels'], predictions,
            ['Anger', 'Happy', 'Neutral', 'Sad'], savedir=f'images/pl{pred_len}', normalize=False
        )

        # Recall score
        all_recall.append(recall_score(result['true_labels'], predictions, average='macro'))

    print('Average UWR:', np.mean(all_recall * 100))