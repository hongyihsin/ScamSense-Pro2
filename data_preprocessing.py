import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. 讀取資料集 (選取計畫書中針對的 PortScan 檔案)
file_path = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
print(f"正在讀取檔案: {file_path}...")
df = pd.read_csv(file_path)

# 2. 移除欄位名稱開頭的空白 (原始資料常有空白，這步很重要)
df.columns = df.columns.str.strip()

# 3. 處理異常值 (Infinity) 與 缺失值 (NaN) 
# 將所有的 Infinity 替換為 NaN，然後統一刪除含有 NaN 的列
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print(f"✅ 清理完成，剩餘樣本數: {len(df)}")

# 4. 標籤轉換：將文字轉為數字 [cite: 38-40, 181-183]
# 0 = Normal Traffic (BENIGN), 1 = Attack Traffic (PortScan)
df['Label'] = df['Label'].map({'BENIGN': 0, 'PortScan': 1})

# 5. 選取關鍵特徵 (根據計畫書提到的 Flow Duration 等) [cite: 37, 180]
# 這裡先選取幾個核心特徵進行示範，你可以根據需要增加
features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packet Length Max']
X = df[features]
y = df['Label']

# 6. 特徵縮放：採用 Min-Max Normalization 
# 使所有數值落於 [0, 1] 之間，這對 DNN 模型訓練非常重要
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_final = pd.DataFrame(X_scaled, columns=features)

# 7. 檢查結果
print("\n--- 前處理結果摘要 ---")
print(X_final.head())
print("\n標籤分布:")
print(y.value_counts())

# 8. 儲存清理後的資料 (方便第二週訓練模型使用)
X_final['Label'] = y.values
X_final.to_csv('cleaned_portscan_data.csv', index=False)
print("\n✅ 清理後的資料已儲存為 'cleaned_portscan_data.csv'")
