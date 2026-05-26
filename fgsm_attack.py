import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# 1. 定義模型架構 (必須與訓練時完全一致)
class IDS_Model(nn.Module):
    def __init__(self, input_size):
        super(IDS_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# 2. 載入模型與資料
df = pd.read_csv('cleaned_portscan_data.csv')
# 只挑選標籤為 1 (攻擊) 的樣本來測試能不能騙過模型
attack_samples = df[df['Label'] == 1].head(100) 
X = torch.FloatTensor(attack_samples.drop('Label', axis=1).values)
y = torch.LongTensor(attack_samples['Label'].values)

model = IDS_Model(X.shape[1])
model.load_state_dict(torch.load('ids_model.pth'))
model.eval()

# 3. FGSM 攻擊函數 [cite: 53-60, 196-203]
def fgsm_attack(data, epsilon, data_grad):
    # 收集梯度的符號 [cite: 55-56, 198-199]
    sign_data_grad = data_grad.sign()
    # 擾動數據：往梯度反方向移動 (對抗性微擾) [cite: 76, 219]
    perturbed_data = data + epsilon * sign_data_grad
    # 加入物理約束：確保數值在 [0, 1] 之間 (計畫書要求) 
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# 4. 執行攻擊實驗
epsilon = 0.1 # 擾動強度 [cite: 72, 215]
X.requires_grad = True # 開啟追蹤梯度

outputs = model(X)
loss = nn.CrossEntropyLoss()(outputs, y)
model.zero_grad()
loss.backward()

# 生成對抗樣本
data_grad = X.grad.data
perturbed_data = fgsm_attack(X, epsilon, data_grad)

# 測試攻擊後的結果
with torch.no_grad():
    new_outputs = model(perturbed_data)
    _, predicted = torch.max(new_outputs, 1)
    # 計算有多少原本是 1 的樣本現在被誤判為 0 [cite: 16, 78, 159, 221]
    fooled_count = (predicted == 0).sum().item()

print(f"🔥 擾動強度 Epsilon: {epsilon}")
print(f"🎯 成功騙過模型的樣本數: {fooled_count} / 100")
