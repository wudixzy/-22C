import pandas as pd

# 读取比特币CSV文件
data_bchain = pd.read_csv("BCHAIN-MKPRU.csv")

# 确保 Date 列为 datetime 类型
data_bchain['Date'] = pd.to_datetime(data_bchain['Date'], format='%m/%d/%y')

# 将日期格式化为 "年-月-日" 格式
data_bchain['Date'] = data_bchain['Date'].dt.strftime('%Y-%m-%d')

# 检查结果
print("修正后的前几行数据：")
print(data_bchain.head())

# 保存修正后的数据到新的CSV文件
output_file_path = "processed_BCHAIN-MKPRU.csv"
data_bchain.to_csv(output_file_path, index=False)

print(f"修正后的数据已成功保存到 {output_file_path}")