import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=400, hidden1_dropout_prob=0.2, hidden2_dropout_prob=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc1_drop = nn.Dropout(p=hidden1_dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(p=hidden2_dropout_prob)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        return x