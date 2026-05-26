import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# 1. 定義模型架構 (與之前一致)
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

# 2. 載入原始資料
df = pd.read_csv('cleaned_portscan_data.csv')
X_raw = torch.FloatTensor(df.drop('Label', axis=1).values)
y_raw = torch.LongTensor(df['Label'].values)

# 3. 產生「對抗性樣本」作為訓練補強
print("🧬 正在生成對抗性樣本以進行防禦強化...")
X_adv_list = []
epsilon = 0.1

# 為了節省時間，我們針對一部分攻擊樣本生成對抗資料
attack_indices = (y_raw == 1).nonzero(as_tuple=True)[0][:50000] 
X_attack = X_raw[attack_indices].clone().detach().requires_grad_(True)
y_attack = y_raw[attack_indices]

# 載入舊模型來產生攻擊方向
model = IDS_Model(X_raw.shape[1])
model.load_state_dict(torch.load('ids_model.pth'))
outputs = model(X_attack)
loss = nn.CrossEntropyLoss()(outputs, y_attack)
model.zero_grad()
loss.backward()

# 生成對抗樣本並加入物理約束 [cite: 58-60, 201-203]
X_adv = X_attack + epsilon * X_attack.grad.data.sign()
X_adv = torch.clamp(X_adv, 0, 1).detach()

# 4. 合併資料：原始資料 + 對抗樣本 
X_train_final = torch.cat([X_raw, X_adv], dim=0)
y_train_final = torch.cat([y_raw, y_attack], dim=0)

# 5. 重新訓練模型 [cite: 67, 210]
print("🏋️ 正在執行對抗訓練 (Adversarial Training)...")
model_def = IDS_Model(X_raw.shape[1])
optimizer = optim.Adam(model_def.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model_def(X_train_final)
    loss = criterion(outputs, y_train_final)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# 6. 儲存強化後的模型 [cite: 63, 206]
torch.save(model_def.state_dict(), 'ids_model_defended.pth')
print("✅ 防禦強化模型已儲存為 ids_model_defended.pth")
