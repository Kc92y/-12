import pandas as pd

# 文件路徑
file_paths = [
    '/Users/youkaiqi/Desktop/人工智慧導論/train.txt',
    '/Users/youkaiqi/Desktop/人工智慧導論/val.txt',
    '/Users/youkaiqi/Desktop/人工智慧導論/test.txt'
]

# 初始化資料框列表
dataframes = []

# 讀取所有資料檔案並合併
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 將檔案內容轉換為字典格式
        data = {'id': [], 'text': [], 'label': []}
        for idx, line in enumerate(lines, start=1):
            text, label = line.strip().split(';')
            data['id'].append(idx)
            data['text'].append(text)
            data['label'].append(label)
        # 轉換為DataFrame
        df = pd.DataFrame(data)
        dataframes.append(df)

# 合併所有資料
combined_data = pd.concat(dataframes, ignore_index=True)

# 建立以情緒為分類的csv檔案
for emotion in combined_data['label'].unique():
    emotion_data = combined_data.copy()
    emotion_data[emotion] = (emotion_data['label'] == emotion).astype(int)
    output = emotion_data[['id', 'text', emotion]]
    output.to_csv(f'/Users/youkaiqi/Desktop/人工智慧導論/Split/{emotion}.csv', index=False)

# 檔案匯出完成
output_files = [f'/Users/youkaiqi/Desktop/人工智慧導論/Split/{emotion}.csv' for emotion in combined_data['label'].unique()]
print(output_files)
