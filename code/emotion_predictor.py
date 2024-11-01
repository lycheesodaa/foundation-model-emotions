from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float, Bool
from sklearn.metrics import recall_score
from tqdm import tqdm
from triton.language import dtype
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


class EmotionForecastModel(nn.Module):
    def __init__(
            self,
            prediction_length: int,
            target_dim: int,
            feat_dynamic_real_dim: int,
            past_feat_dynamic_real_dim: int,
            context_length: int,
            num_emotions: int,
            hidden_dim: int = 256,
            module_kwargs: Optional[dict] = None,
            module: Optional[MoiraiModule] = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-base"),
            patch_size: int | str = "auto",
            num_samples: int = 100,
    ):
        super().__init__()

        # Initialize MOIRAI base model
        self.moirai = MoiraiForecast(
            prediction_length=prediction_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            context_length=context_length,
            module_kwargs=module_kwargs,
            module=module,
            patch_size=patch_size,
            num_samples=num_samples
        )

        # Emotion classification head
        self.classification_head = nn.Sequential(
            nn.Linear(prediction_length * target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

    def forward(
            self,
            past_target: Float[torch.Tensor, "batch past_time tgt"],
            past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
            past_is_pad: Bool[torch.Tensor, "batch past_time"],
            feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
            observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
            past_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
            past_observed_feat_dynamic_real: Optional[Float[torch.Tensor, "batch past_time past_feat"]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get predictions from MOIRAI
        forecasts = self.moirai(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,

            # TODO potentially change
            num_samples=1
        )

        # Flatten the forecasts for classification
        batch_size = forecasts.shape[0]
        flattened_forecasts = forecasts.reshape(batch_size, -1)

        # Get emotion predictions
        emotion_logits = self.classification_head(flattened_forecasts)

        return forecasts, emotion_logits


class EmotionPredictor:
    def __init__(
            self,
            model: EmotionForecastModel,
            device: torch.device,
            batch_size: int = 16,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

    def prepare_input_tensors(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for the MOIRAI model with proper masking of zero values.

        Args:
            features: Input features tensor of shape (batch_size, sequence_length, feature_dim)

        Returns:
            tuple of:
                - past_target: The feature values
                - past_observed_target: Boolean mask indicating non-zero values
                - past_is_pad: Boolean mask indicating padding at sequence level
        """
        # The features tensor will be used directly as past_target
        past_target = features

        # Create mask for non-zero values across all feature dimensions
        # If any feature dimension is non-zero, we consider that timestep as observed
        # past_observed_target = (features != 0.0)
        past_observed_target = torch.ones_like(features, dtype=torch.bool)

        # For sequence-level padding, we consider a timestep padded if all features are zero
        # Shape: (batch_size, sequence_length)
        past_is_pad = (features.sum(dim=2) == 0)

        return past_target, past_observed_target, past_is_pad

    def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, total=len(train_loader)):
            features, emotion_labels = [b.to(self.device) for b in batch]

            # Prepare input tensors
            past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

            optimizer.zero_grad()

            # Forward pass
            _, emotion_logits = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )

            # Calculate loss
            loss = criterion(emotion_logits, emotion_labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(
            self,
            val_loader: torch.utils.data.DataLoader,
            criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_UWR = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                features, emotion_labels = batch[0].to(self.device), batch[1].to(self.device)

                # Prepare input tensors
                past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

                # Forward pass
                _, emotion_logits = self.model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad
                )

                # Calculate loss
                loss = criterion(emotion_logits, emotion_labels)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(emotion_logits.data, 1)
                total_UWR += recall_score(emotion_labels, predicted, average='macro')

        return total_loss / len(val_loader), total_UWR / len(val_loader)

    def predict(
            self,
            features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        with torch.no_grad():
            # Prepare input tensors
            past_target, past_observed_target, past_is_pad = self.prepare_input_tensors(features)

            # Get predictions
            forecasts, emotion_logits = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad
            )

            emotion_probs = torch.softmax(emotion_logits, dim=1)

        return forecasts.cpu(), emotion_probs.cpu()