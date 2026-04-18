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

output_path = os.path.join(desktop_path, "ICData_清洗后.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n处理完成")

import matplotlib.pyplot as plt
# 筛选：仅 刷卡类型=0
data = df[df["刷卡类型"] == 0]
hours = data["hour"].values  # 转 numpy 数组

# 1. numpy 布尔索引统计
total = len(hours)
morning_before = np.sum(hours < 7)    # 早峰前 <7
night = np.sum(hours >= 22)           # 深夜 >=22

# 2. 计算百分比
pct_morning = (morning_before / total) * 100
pct_night = (night / total) * 100

# 打印结果
print("=" * 50)
print("【(a) 早晚时段刷卡量统计（numpy 实现）】")
print(f"全天总刷卡量（刷卡类型=0）：{total}")
print(f"早峰前时段（hour < 7）：{morning_before} 次，占比 {pct_morning:.2f}%")
print(f"深夜时段（hour ≥ 22）：{night} 次，占比 {pct_night:.2f}%")
print("=" * 50)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False

# 统计每个小时的刷卡量
hour_counts = np.bincount(hours, minlength=24)
x = np.arange(24)

# 绘图
colors = ["#87CEEB"] * 24
for h in range(24):
    if h < 7 or h >= 22:
        colors[h] = "#FF7F50"
plt.figure(figsize=(12, 6))
bars = plt.bar(x, hour_counts, color=colors, width=0.7)
plt.title("24小时公交刷卡量分布", fontsize=16, pad=15)
plt.xlabel("小时（0~23）", fontsize=12)
plt.ylabel("刷卡量（次）", fontsize=12)
plt.xticks(np.arange(0, 24, 2), fontsize=10)  # 步长2
plt.grid(axis="y", linestyle="--", alpha=0.6)  # 水平网格线
plt.bar([0], [0], color="#87CEEB", label="普通时段")
plt.bar([0], [0], color="#FF7F50", label="早峰前(<7) / 深夜(≥22)")
plt.legend(fontsize=11)

# 保存图片到桌面
save_path = os.path.join(desktop_path, "hour_distribution.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()


