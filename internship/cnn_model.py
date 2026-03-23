import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn

df = pd.read_csv("./Dataset/breathing_windows_final.csv")

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["flow_window"] = df["flow_window"].apply(parse_list)
df["thor_window"] = df["thor_window"].apply(parse_list)

# encoding labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


participants = df["patient"].unique()
all_preds = []
all_true = []

for test_patient in participants:
    print("Testing on:", test_patient)
    train_X, train_y = [], []
    test_X, test_y = [], []

    for _, row in df.iterrows():
        flow = row["flow_window"]
        thor = row["thor_window"]

        if len(flow) != 30 or len(thor) != 30:
            continue

        sample = np.stack([flow, thor], axis=0)

        if row["patient"] == test_patient:
            test_X.append(sample)
            test_y.append(row["label_encoded"])
        else:
            train_X.append(sample)
            train_y.append(row["label_encoded"])

    # converting to tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)
    model = SimpleCNN(num_classes=len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

    # testing
    model.eval()
    with torch.no_grad():
        outputs = model(test_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())
        all_true.extend(test_y.numpy())


# metrics
accuracy = accuracy_score(all_true, all_preds)
precision = precision_score(all_true, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_true, all_preds, average="weighted", zero_division=0)
cm = confusion_matrix(all_true, all_preds)

print("\nFinal Results:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Confusion Matrix:")
print(cm)