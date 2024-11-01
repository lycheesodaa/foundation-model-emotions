import torch
import torch.nn as nn

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

