import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 讀取清理後的資料
df = pd.read_csv('cleaned_portscan_data.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# 2. 切分與轉換張量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 3. DNN 模型架構 (增加神經元數量提升表達力)
class IDS_Model(nn.Module):
    def __init__(self, input_size):
        super(IDS_Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # 防止過擬合
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# 4. 初始化
model = IDS_Model(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 強化訓練 (增加到 100 次迭代)
print("🚀 開始深度訓練模式...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        # 每 10 次顯示一次目前的準確率
        model.eval()
        with torch.no_grad():
            _, predicted = torch.max(model(X_test), 1)
            acc = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Test Acc: {acc*100:.2f}%')

# 6. 儲存最終模型 
torch.save(model.state_dict(), 'ids_model.pth')
print("✅ 高精度模型已訓練完成！")
