import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress tracking
from sklearn.model_selection import KFold
import os
from torch.nn import MSELoss


class TrajectoryDataset(Dataset):
    def __init__(self, dataframe, window_length=100):
        # Perform the custom transformation
        sliced_df = self.custom_transformation(
            dataframe.to_numpy(), window_length=window_length)
        self.data = torch.tensor(sliced_df, dtype=torch.float32)

    def __len__(self):
        # Return the number of trajectories
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Get the trajectory at the given index
        return self.data[idx]

    def custom_transformation(self, dataframe_array, window_length):
        num_rows, num_cols = dataframe_array.shape
        window_length += 1  # get one more column as targets

        # Preallocate memory for the slices
        sliced_data = np.lib.stride_tricks.sliding_window_view(
            dataframe_array, window_shape=(window_length,), axis=1
        )
        # Reshape into a flat 2D array for DataFrame-like output
        sliced_data = sliced_data.reshape(-1, window_length)

        return sliced_data


# Implement your model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout5 = nn.Dropout(0.3)

        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.dropout5(out)

        out = self.fc_out(out)
        return out


train_path = os.path.join('train.csv')
val_path = os.path.join('val.csv')
test_path = os.path.join('test.csv')

train_df = pd.read_csv(train_path, header=0).drop('ids', axis=1)
val_df = pd.read_csv(val_path, header=0).drop('ids', axis=1)
test_df = pd.read_csv(test_path, header=0).drop('ids', axis=1)

device = torch.device('cuda')

window_length = 100
batch_size = 64 
k_folds = 5
num_epochs = 5
learning_rate = 0.0003
input_size = window_length
hidden_size = 256 
output_size = 1

dataset = TrajectoryDataset(dataframe=train_df, window_length=window_length)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

model = MLP(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}/{k_folds}')

    model.apply(
        lambda m: m.reset_parameters()
        if hasattr(m, 'reset_parameters') else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=0, threshold=1e-6
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    for epoch in tqdm(
        range(num_epochs), desc=f"Fold {fold + 1} - Epochs",
        unit="epoch", leave=True
    ):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            inputs = data[:, :-1].to(device)
            targets = data[:, -1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f'Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], '
            f'Training Loss: {running_loss / len(train_loader):.6f}'
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[:, :-1].to(device)
                targets = data[:, -1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Fold {fold + 1}, Validation Loss: {avg_val_loss:.6f}')

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Learning Rate after Epoch {epoch + 1}: {current_lr}')

        scheduler.step(avg_val_loss)

    fold_results.append(avg_val_loss)

print(
    f'Average Validation Loss across {k_folds} '
    f'folds: {np.mean(fold_results):.6f}')

train_set = torch.tensor(train_df.values[:, :].astype(np.float32)).to(device)
val_set = torch.tensor(val_df.values[:, :].astype(np.float32)).to(device)
test_set = torch.tensor(val_df.values[:, :].astype(np.float32)).to(device)

points_to_predict = val_set.shape[1]


# Autoregressive prediction function
def autoregressive_predict(
    model, input_maxtrix, prediction_length=points_to_predict
):
    model.eval()
    output_matrix = torch.empty(
        input_maxtrix.shape[0], 0, device=device)
    current_input = input_maxtrix.to(device)
    with torch.no_grad():
        for _ in range(prediction_length):
            next_pred = model(current_input)
            output_matrix = torch.cat((output_matrix, next_pred), dim=1)
            current_input = torch.cat((current_input[:, 1:], next_pred), dim=1)
    return output_matrix


initial_input = train_set[:, -window_length:]
full_trajectories = autoregressive_predict(model, initial_input,)
full_trajectories = full_trajectories.to(device)
val_set = val_set.to(device)

mse_loss = MSELoss()

mse = mse_loss(full_trajectories, val_set)

print(f'Autoregressive Validation MSE (using torch): {mse.item():.4f}')

row_idx = 0
initial_input = val_set[row_idx, :window_length].unsqueeze(0)

predicted_trajectory = autoregressive_predict(model, initial_input)
predicted_trajectory = predicted_trajectory.squeeze().cpu().numpy()

actual_trajectory = val_set[row_idx].cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(
    range(len(actual_trajectory)), actual_trajectory,
    label="Actual Trajectory", color='blue', marker='o')
plt.plot(
    range(len(actual_trajectory)), predicted_trajectory,
    label="Predicted Trajectory", color='red', linestyle='--', marker='x')
plt.title(f"Actual vs Predicted Trajectory (Row {row_idx})")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig(f'predictions_plot.png')
plt.close()

# Generate predictions for all the validation dataset
val_predictions = []
for i in tqdm(range(val_set.shape[0])):
    initial_input = val_set[i, :window_length].unsqueeze(0)
    predicted_trajectory = autoregressive_predict(model, initial_input)
    val_predictions.append(predicted_trajectory.squeeze().cpu().numpy())

# Generate predictions for all the testing dataset
test_predictions = []
for i in tqdm(range(test_set.shape[0])):
    initial_input = test_set[i, :window_length].unsqueeze(0)
    predicted_trajectory = autoregressive_predict(model, initial_input)
    test_predictions.append(predicted_trajectory.squeeze().cpu().numpy())

val_predictions_np = np.array(val_predictions)
test_predictions_np = np.array(test_predictions)

val_predictions_tensor = torch.tensor(val_predictions_np, dtype=torch.float32).to(device)
test_predictions_tensor = torch.tensor(test_predictions_np, dtype=torch.float32).to(device)

print(f'Validation Predictions Tensor Shape: {val_predictions_tensor.shape}')
print(f'Test Predictions Tensor Shape: {test_predictions_tensor.shape}')

for idx in range(3):  # Indices 0, 1, 2
    plt.figure(figsize=(4, 4))
    plt.plot(val_predictions_tensor[idx, :].cpu().numpy(), color='black', linewidth=3, linestyle='-')
    plt.title(f'Trajectory {idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    # Save each figure as PNG with dpi 200
    plt.savefig(f'trajectory_{idx}.png', dpi=200)
    plt.close()


def generate_submissions_v2(
    pred_val_tensor, pred_test_tensor, original_val_path, original_test_path
):
    original_val_df = pd.read_csv(original_val_path)
    original_test_df = pd.read_csv(original_test_path)

    assert (
        pred_val_tensor.shape[0] * pred_val_tensor.shape[1] ==
        original_val_df.shape[0] * (original_val_df.shape[1] - 1)
    )
    assert (
        pred_test_tensor.shape[0] * pred_test_tensor.shape[1] ==
        original_test_df.shape[0] * (original_test_df.shape[1] - 1)
    )

    ids = []
    values = []

    for col_idx, col in enumerate(original_val_df.columns[1:]):
        for row_idx, _ in enumerate(original_val_df[col]):
            ids.append(str(f"{col}_traffic_val_{row_idx}"))
            values.append(float(pred_val_tensor[row_idx, col_idx]))

    for col_idx, col in enumerate(original_test_df.columns[1:]):
        for row_idx, _ in enumerate(original_test_df[col]):
            ids.append(str(f"{col}_traffic_test_{row_idx}"))
            values.append(float(pred_test_tensor[row_idx, col_idx]))

    submissions_df = pd.DataFrame({
        "ids": ids,
        "value": values
    })

    assert submissions_df.shape[1] == 2
    assert (
        submissions_df.shape[0] ==
        (original_val_df.shape[0] * (original_val_df.shape[1] - 1)) +
        (original_test_df.shape[0] * (original_test_df.shape[1] - 1))
    )
    assert "ids" in submissions_df.columns
    assert "value" in submissions_df.columns
    assert submissions_df['value'].isnull().sum() == 0

    submissions_df.to_csv('submissions.csv', index=False)


generate_submissions_v2(val_predictions_tensor, test_predictions_tensor, 'val.csv', 'test.csv')
