import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import recall_score, confusion_matrix
from tqdm import tqdm
from uni2ts.model.moirai import MoiraiModule

from utils.utils import prepare_data, confusion_matrix_plot
from models.emotion_predictor import EmotionForecastModel, EmotionPredictor
from data_provider.emotion_dataset import EmotionDataset


def train_emotion_forecast_model(features, labels, speaker_ids, prediction_length=24, batch_size=16, num_epochs=50,
                                 patience=10, ckpt_path='checkpoints/', accumulation_steps=1):
    # Model parameters
    target_dim = features.shape[2]  # Feature dimension
    context_length = features.shape[1]  # Sequence length
    num_emotions = len(set(labels))

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # Initialize model
    model = EmotionForecastModel(
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=0,  # adjust if you have dynamic features
        past_feat_dynamic_real_dim=0,  # adjust if you have past dynamic features
        context_length=context_length,
        num_emotions=num_emotions,
        hidden_dim=256,
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
        patch_size=8 # this argument is necessary to override 'auto' so MOIRAI processes only CTX, not CTX+PDT
    )

    # Setup cross-validation
    logo = LeaveOneGroupOut()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
            EmotionDataset(features[train_subset], labels[train_subset], verbose=True),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            EmotionDataset(features[val_subset], labels[val_subset]),
            batch_size=batch_size
        )
        test_loader = DataLoader(
            EmotionDataset(features[test_idx], labels[test_idx]),
            batch_size=batch_size
        )

        # Initialize training components
        predictor = EmotionPredictor(model, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = predictor.train_epoch(train_loader, optimizer, criterion, accumulation_steps)

            # Validate
            val_loss, val_uwr = predictor.validate(val_loader, criterion)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val UWR = {val_uwr:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), Path(ckpt_path) / f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break


        # Test loop - get predictions
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(Path(ckpt_path) / f'best_model_speaker_{speaker_ids[test_idx[0]]}.pt'))
        predictor = EmotionPredictor(model, device=device)

        all_forecasts = []
        all_probs = []
        all_labels = []

        for batch in tqdm(test_loader, total=len(test_loader)):
            features_batch, labels_batch = [b.to(device) for b in batch]
            forecasts, probs = predictor.predict(features_batch)
            all_forecasts.append(forecasts)
            all_probs.append(probs)
            all_labels.extend(labels_batch.cpu().numpy())

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
    batch_size = 8
    accumulation_steps = 4

    # Prepare your data - Moirai uses the mean only data, so 179 features
    features, labels, speaker_ids = prepare_data(directory='Files/UF_His_data/step_1_mean', mean_only=True)

    # Train model and get results
    results = train_emotion_forecast_model(features, labels, speaker_ids,
                                           prediction_length=pred_len,
                                           batch_size=batch_size, accumulation_steps=accumulation_steps)

    all_recall = []

    # Print results for each speaker
    for speaker, result in results.items():
        predictions = np.argmax(result['probabilities'], axis=1)
        accuracy = np.mean(predictions == result['true_labels'])
        uwr = recall_score(result['true_labels'], predictions, average='macro')
        print(f"\nResults for Speaker {speaker}:")
        print(f"Accuracy - {accuracy:.4f}")
        print(f"UWR - {uwr:.4f}")

        # Confusion matrix
        try:
            print("Confusion Matrix:")
            print(confusion_matrix(result['true_labels'], predictions))
            confusion_matrix_plot(
                str(speaker), result['true_labels'], predictions,
                ['Anger', 'Happy', 'Neutral', 'Sad'], savedir=f'Images/pl{pred_len}', normalize=False
            )
        except Exception as e:
            print('Error plotting confusion matrix:', e)

        # Recall score
        all_recall.append(uwr)

    print('Average UWR:', np.mean(all_recall * 100))