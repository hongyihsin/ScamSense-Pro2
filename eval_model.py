import torch
import torch.nn as nn
import pandas as pd

# 定義跟訓練時一樣的架構
class IDS_Model(nn.Module):
    def __init__(self, input_size):
        super(IDS_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 1. 載入資料與模型
df = pd.read_csv('cleaned_portscan_data.csv')
X = torch.FloatTensor(df.drop('Label', axis=1).values)
y = torch.LongTensor(df['Label'].values)

model = IDS_Model(X.shape[1])
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

# 2. 執行預測
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / y.size(0)

print(f"🎯 當前模型準確率: {accuracy*100:.2f}%")
