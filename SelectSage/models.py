import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBinaryClassifer(nn.Module):
    def __init__(self, in_size, mlp_h1_size, mlp_h2_size):
        super().__init__()
        self.W1 = nn.Linear(in_size, mlp_h1_size)
        self.W2 = nn.Linear(mlp_h1_size, mlp_h2_size)
        self.W3 = nn.Linear(mlp_h2_size, 1)

    def forward(self, x):
        x = F.relu(self.W1(x))
        x = F.relu(self.W2(x))
        x = self.W3(x)
        return x
    
    def predict(self, x):
        pred = torch.sigmoid(self.forward(x))
        result = []
        for out in pred:
            if out <= 0.5:
                result.append(0)
            else:
                result.append(1)
        return torch.tensor(result)

class MLPMultiClassifer(nn.Module):
    def __init__(self, in_size, mlp_h1_size, mlp_h2_size, num_classes):
        super().__init__()
        self.W1 = nn.Linear(in_size, mlp_h1_size)
        self.W2 = nn.Linear(mlp_h1_size, mlp_h2_size)
        self.W3 = nn.Linear(mlp_h2_size, num_classes)

    def forward(self, x):
        x = F.relu(self.W1(x))
        x = F.relu(self.W2(x))
        x = self.W3(x)
        return x
    
    def predict(self, x):
        func = nn.Softmax(dim=1)
        pred = func(self.forward(x))
        result = []
        for out in pred:
            result.append(out.argmax())
        return torch.tensor(result)
