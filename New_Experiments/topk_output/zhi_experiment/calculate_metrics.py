import pandas as pd

df = pd.read_csv('New_Experiments/topk_output/zhi_experiment/0.7_0.9_0.5_uniform_synthetic_21-02-10.csv')

# 新增一個指標 avgNewRecourseCost / avgOriginalRecourseCost
df['RatioofEffort'] = df['avgNewRecourseCost'] / df['avgOriginalRecourseCost']
df['model_shift'] = df['model_shift'].str.replace('tensor\\(', '', regex=True).str.replace('\\)', '', regex=True).astype(float)
# 計算每個指標的平均值和標準差，包括新指標
statistics = df.agg(['mean', 'std'])

# 查看結果
print(df.head())
print(statistics)