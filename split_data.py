import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 定義情緒類別及檔案路徑
emotions = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
base_path = '/Users/youkaiqi/Desktop/人工智慧導論/BERT/class10(KFold)'

# 設定分層 KFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 處理每個情緒檔案
for emotion in emotions:
    # 讀取CSV檔案
    file_path = f"/Users/youkaiqi/Desktop/人工智慧導論/Split/{emotion}.csv"
    df = pd.read_csv(file_path)

    # 分層切割
    for fold_num, (train_index, test_index) in enumerate(skf.split(df, df[emotion])):
        # 建立資料夾結構
        emotion_folder = os.path.join(base_path, emotion, f"{emotion}_{fold_num}")
        os.makedirs(emotion_folder, exist_ok=True)

        # 建立訓練集和測試集
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        # 存儲訓練集和測試集
        train_file_path = os.path.join(emotion_folder, f"train_{emotion}_{fold_num}.csv")
        test_file_path = os.path.join(emotion_folder, f"test_{emotion}_{fold_num}.csv")

        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)

print("資料切割完成並已存入指定路徑。")
