import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# -------------------------------
# SEQUENCE BUILDER
# -------------------------------
def create_sequences(data, seq_length=14):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


# -------------------------------
# LSTM MODEL
# -------------------------------
class OzoneLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=7):
        super(OzoneLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train_lstm(df, target_col="Ozone (µg/m³)", seq_length=60, epochs=150):

    data = df[target_col].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled, seq_length)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = OzoneLSTM(hidden_size=128)  # stronger model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    batch_size = 16

    for epoch in range(epochs):
        model.train()

        permutation = torch.randperm(X_train.size(0))

        epoch_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    return model, scaler, X_test, y_test

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate_lstm(model, scaler, X_test, y_test):

    model.eval()

    with torch.no_grad():
        preds = model(X_test).numpy()

    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test.numpy())

    preds = preds.flatten()
    y_test = y_test.flatten()

    mae = np.mean(np.abs(preds - y_test))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return preds, y_test, mae, r2