import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Define the model
class SimplePerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimplePerceptron, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def load_data():
    data = pd.read_csv("data/token_features.csv")
    X = data.drop(columns=["target"]).values
    y = data["target"].values
    return X, y

def train_model(X, y):
    input_size = X.shape[1]
    output_size = 2

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimplePerceptron(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):  # Training for 10 epochs
        for data, label in dataloader:
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models/perceptron.pth")

if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)