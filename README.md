# 4.16
import pandas as pd
import numpy as np
import os

# 1. 定位桌面文件(手动处理，否则默认在pycharm所在文件夹种，读取出错)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "ICData.csv")

# 2. 正确读取（改为逗号分隔 sep=","，解决列名识别错误）
df = pd.read_csv(file_path, sep=",")

# 打印前5行和基本信息
print(df.head())
print("\n数据集基本信息")
print(f"行数：{df.shape[0]}，列数：{df.shape[1]}")
print("\n各列数据类型：")
print(df.dtypes)

# 3. 时间解析（现在能正确识别「交易时间」列）
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["hour"] = df["交易时间"].dt.hour
print("\n时间处理完成")
print(df[["交易时间", "hour"]].head())

# 4. 构造搭乘站点数并采取了直接删除异常数据策略
df["ride_stops"] = abs(df["下车站点"] - df["上车站点"])
delete_num = (df["ride_stops"] == 0).sum()
df = df[df["ride_stops"] != 0].reset_index(drop=True)
print(f"\n异常记录处理")
print(f"删除 ride_stops=0 的行数：{delete_num}")

# 5. 缺失值检查与处理
print("\n各列缺失值数量")
print(df.isnull().sum())
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
output_path = os.path.join(desktop_path, "ICData_清洗后.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n处理完成")
