import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.models.lstm_model import LSTMModel
from sklearn.preprocessing import StandardScaler


def create_sequences(X, y, seq_length=5):

    X_seq, y_seq = [], []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])

    return np.array(X_seq), np.array(y_seq)


def train_lstm(df, target_col):

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    # -------------------------------
    # SCALING (VERY IMPORTANT)
    # -------------------------------
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences(X, y)

    # Train-test split
    split = int(0.8 * len(X_seq))

    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Model
    model = LSTMModel(input_size=X.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(50):

        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

        # inverse transform
        preds = scaler_y.inverse_transform(preds)
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return mae, r2, preds, y_test