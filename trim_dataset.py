import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 讀取妳從 Kaggle 下載下來的巨大 CSV 檔 (請把檔名改成妳下載的實際檔名)
large_file_path = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
print("正在讀取巨大檔案，請稍候...")
df = pd.read_csv(large_file_path)

# 為了防止欄位名稱前後有空格，先幫所有欄位名稱去空格
df.columns = df.columns.str.strip()

# 2. 定義妳們系統需要的 5 個核心特徵與標籤
target_features = [
    'Destination Port', 
    'Flow Duration', 
    'Total Fwd Packets', 
    'Total Backward Packets', 
    'Fwd Packet Length Max'
]

# 3. 確保這些欄位存在，並只抽取前 500 筆樣本
if all(feat in df.columns for feat in target_features):
    # 只抓需要的欄位
    sub_df = df[target_features].copy()
    
    # 處理 Label 欄位：將文字 (如 BENIGN, PortScan) 轉為 0 與 1
    if 'Label' in df.columns:
        # 如果包含 PortScan 關鍵字就設為 1，其餘(BENIGN)設為 0
        sub_df['Label'] = df['Label'].apply(lambda x: 1 if 'PortScan' in str(x) else 0)
    else:
        sub_df['Label'] = 0 # 防呆用
        
    # 4. 執行全球網絡標準 Min-Max 特徵縮放 [0, 1]
    scaler = MinMaxScaler()
    sub_df[target_features] = scaler.fit_transform(sub_df[target_features])
    
    # 只取前 500 筆，檔案直接縮小到幾 KB 
    demo_df = sub_df.head(500)
    
    # 5. 輸出成完美的 Demo 專用小檔案
    demo_df.to_csv("my_custom_demo_traffic.csv", index=False)
    print("✨ 恭喜！完美加工過的測試檔案已生成：`my_custom_demo_traffic.csv`")
    print(demo_df.head())
else:
    print("❌ 錯誤：這個 CSV 檔案中找不到對應的資安特徵欄位，請檢查檔案內容！")