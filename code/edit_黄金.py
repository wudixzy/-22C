import pandas as pd
import numpy as np

# 读取两个CSV文件
data_bchain = pd.read_csv("BCHAIN-MKPRU.csv")
data_gold = pd.read_csv("data_GOLD_original.csv")

# 确保 Date 列为 datetime 类型，并设置为索引
data_bchain['Date'] = pd.to_datetime(data_bchain['Date'])
data_bchain.set_index('Date', inplace=True)

data_gold['Date'] = pd.to_datetime(data_gold['Date'])
data_gold.set_index('Date', inplace=True)

# 假设 data_bchain 是完整数据，data_gold 是有缺失值的数据
data_complete = data_bchain.copy()
data_with_gaps = data_gold.copy()

# 创建一个新的 DataFrame 用于保存最终结果
result_df = data_with_gaps.copy()

# 插入缺失日期并排序
full_date_range = pd.date_range(start=data_complete.index.min(), end=data_complete.index.max(), freq='D')
result_df = result_df.reindex(full_date_range)

# 标记缺失值
result_df['is_interpolated'] = result_df['USD (PM)'].isnull()

# 使用线性插值填充缺失值（这里假设 'USD (PM)' 是要处理的列名）
# 因为你提到每次缺失都是连续出现两个，我们可以确保前后有有效的数据点来进行插值
result_df['USD (PM)'] = result_df['USD (PM)'].interpolate(method='linear', limit_direction='both')

# 处理连续缺失值，确保每次缺失都是两个连续的数据点
for idx in result_df.index:
    if result_df.loc[idx, 'is_interpolated']:
        # 检查是否连续两个缺失值都被正确插值
        next_idx = result_df.index.get_loc(idx) + 1
        if next_idx < len(result_df) and result_df.iloc[next_idx]['is_interpolated']:
            # 计算前后两个有效值的平均值
            prev_valid = result_df.loc[:idx].dropna(subset=['USD (PM)'])['USD (PM)'].iloc[-1]
            next_valid = result_df.loc[idx:].dropna(subset=['USD (PM)'])['USD (PM)'].iloc[0]
            avg_value = (prev_valid + next_valid) / 2
            
            # 设置这两个位置的值为平均值
            result_df.loc[idx, 'USD (PM)'] = avg_value
            result_df.iloc[next_idx, result_df.columns.get_loc('USD (PM)')] = avg_value

# 重置索引以便查看结果
result_df.reset_index(inplace=True)
result_df.rename(columns={'index': 'Date'}, inplace=True)

# 保存结果到新的CSV文件
output_file_path = "processed_data_gold.csv"
result_df.to_csv(output_file_path, index=False)

print(f"处理后的数据已成功保存到 {output_file_path}")