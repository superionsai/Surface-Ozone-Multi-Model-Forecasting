import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Computes a self-attention distribution over the sequence length.
    Allows the model to focus on specific past hours (e.g., yesterday's peak).
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch, seq_len, hidden_size)
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Multiply weights by LSTM outputs to get the context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        return context, attn_weights

class AdvancedOzoneLSTM(nn.Module):
    """
    Advanced BiLSTM architecture with Temporal Attention and Residual Connections.
    Expects input shape: [batch_size, seq_length, num_features]
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.4):
        super(AdvancedOzoneLSTM, self).__init__()
        
        # Bidirectional LSTM reads the sequence forwards and backwards
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        
        # Hidden size is doubled due to bidirectional output
        lstm_out_size = hidden_size * 2
        
        # Temporal Attention Layer
        self.attention = TemporalAttention(lstm_out_size)
        
        # Fully connected block with Residual skip connection
        self.fc1 = nn.Linear(lstm_out_size, lstm_out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for regression stability
        self.batch_norm = nn.BatchNorm1d(lstm_out_size)
        
        # Final output layer
        self.fc2 = nn.Linear(lstm_out_size, 1)

    def forward(self, x):
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Compute context vector using Attention over the entire sequence
        context, attn_weights = self.attention(lstm_out)  # context: (batch, hidden_size * 2)
        
        # Fully connected residual block
        residual = context
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Add residual skip connection
        out = out + residual
        
        # Batch Norm
        out = self.batch_norm(out)
        
        # Final prediction
        out = self.fc2(out)
        
        return out.squeeze(dim=-1)
