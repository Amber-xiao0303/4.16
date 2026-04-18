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

import seaborn as sns
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "ICData.csv")

df = pd.read_csv(file_path, sep=None, engine="python")
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["hour"] = df["交易时间"].dt.hour
df["ride_stops"] = abs(df["下车站点"] - df["上车站点"])
df = df[df["ride_stops"] != 0].reset_index(drop=True)

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    # 分组计算均值和标准差
    result = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    result.columns = ['线路号', 'mean_stops', 'std_stops']
    # 按均值降序排序
    result = result.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return result

route_stats = analyze_route_stops(df)
print("=" * 60)
print("各线路平均搭乘站点数（前10行）")
print("=" * 60)
print(route_stats.head(10))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 取均值最高的前15条线路
top15 = route_stats.head(15)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top15,
    x='mean_stops',
    y='线路号',
    xerr='std_stops',  # 误差棒=标准差
    capsize=0.3,       # 误差棒端点
    palette='Blues_d'
)

# 图表样式
plt.title('各线路平均搭乘站点数（Top15）', fontsize=14)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(left=0)

save_path = os.path.join(desktop_path, "route_stops.png")
plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "ICData.csv")

df = pd.read_csv(file_path, sep=None, engine="python")
df["交易时间"] = pd.to_datetime(df["交易时间"])
df = df[df["刷卡类型"] == 0].copy().reset_index(drop=True)  # 只算上车刷卡

df['hour'] = df['交易时间'].dt.hour
hour_count = df.groupby('hour').size()
peak_hour = hour_count.idxmax()
peak_volume = hour_count.max()

print(f"高峰小时：{peak_hour:02d}:00 ~ {peak_hour+1:02d}:00，刷卡量：{peak_volume} 次")

# 筛选高峰小时内的所有数据
peak_hour_data = df[df['hour'] == peak_hour].copy()

peak_hour_data['5min_bin'] = peak_hour_data['交易时间'].dt.floor('5min')
max_5min = peak_hour_data.groupby('5min_bin').size().max()
phf5 = peak_volume / (12 * max_5min)

# 找到最大5分钟时间段
# dt.floor('5min')：将交易时间按5分钟向下取整（如20:03归为20:00，确保时间区间准确）
# 生成5分钟时间片标签，用于分组统计
max_5min_time = peak_hour_data.groupby('5min_bin').size().idxmax()
max_5min_str = f"{max_5min_time.strftime('%H:%M')}~{(max_5min_time + pd.Timedelta(minutes=5)).strftime('%H:%M')}"

print(f"最大5分钟刷卡量（{max_5min_str}）：{max_5min} 次")
print(f"PHF5  = {peak_volume} / (12 × {max_5min}) = {phf5:.4f}")

peak_hour_data['15min_bin'] = peak_hour_data['交易时间'].dt.floor('15min')
max_15min = peak_hour_data.groupby('15min_bin').size().max()
phf15 = peak_volume / (4 * max_15min)

# 找到最大15分钟时间段
# 格式化输出时间段（如20:00~20:15）
# PHF15计算公式：高峰小时总刷卡量 ÷（4个15分钟 × 最大15分钟刷卡量）
# dt.floor('15min')：将交易时间按15分钟向下取整，生成15分钟时间片标签
max_15min_time = peak_hour_data.groupby('15min_bin').size().idxmax()
max_15min_str = f"{max_15min_time.strftime('%H:%M')}~{(max_15min_time + pd.Timedelta(minutes=15)).strftime('%H:%M')}"

print(f"最大15分钟刷卡量（{max_15min_str}）：{max_15min} 次")
print(f"PHF15 = {peak_volume} / ( 4 × {max_15min}) = {phf15:.4f}")

df = pd.read_csv(file_path, sep=None, engine="python")
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["ride_stops"] = abs(df["下车站点"] - df["上车站点"])
df = df[df["ride_stops"] != 0].reset_index(drop=True)

# 1. 筛选线路号在 1101 ~ 1120 之间的记录
df_filtered = df[(df["线路号"] >= 1101) & (df["线路号"] <= 1120)].copy()

# 2. 在程序根目录创建 线路驾驶员信息 文件夹
root_dir = os.path.dirname(os.path.abspath(__file__))  # 程序根目录
output_folder = os.path.join(root_dir, "线路驾驶员信息")
os.makedirs(output_folder, exist_ok=True)

# 3. 遍历每条线路，生成去重后的 车辆→驾驶员 文件
unique_routes = sorted(df_filtered["线路号"].unique())

for route in unique_routes:
    # 筛选当前线路数据
    route_data = df_filtered[df_filtered["线路号"] == route]

    # 去重：保留唯一的 车辆编号 → 驾驶员编号 对应关系
    driver_map = route_data[["车辆编号", "驾驶员编号"]].drop_duplicates().sort_values("车辆编号")

    # 文件名：1101.txt
    file_name = f"{route}.txt"
    save_path = os.path.join(output_folder, file_name)

    # 写入txt文件（严格按要求格式）
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"线路号: {route}\n")
        f.write("车辆编号\t驾驶员编号\n")
        for _, row in driver_map.iterrows():
            f.write(f"{int(row['车辆编号'])}\t\t{int(row['驾驶员编号'])}\n")

    # 4. 打印每个文件的生成路径
    print(f"已生成：{save_path}")

print("\n20条线路文件导出成功")

df = pd.read_csv(file_path, sep=None, engine="python")
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["ride_stops"] = abs(df["下车站点"] - df["上车站点"])
df = df[df["ride_stops"] != 0].reset_index(drop=True)
df = df[df["刷卡类型"] == 0].copy()  # 只算有效上车记录

# 统计规则：每条有效记录 = 1人次
top10_driver    = df['驾驶员编号'].value_counts().head(10)
top10_route     = df['线路号'].value_counts().head(10)
top10_stop      = df['上车站点'].value_counts().head(10)
top10_bus       = df['车辆编号'].value_counts().head(10)

# 打印所有排名
print("=" * 70)
print("【Top10 司机（服务人次）】")
print(top10_driver)
print("\n【Top10 线路（服务人次）】")
print(top10_route)
print("\n【Top10 上车站点（服务人次）】")
print(top10_stop)
print("\n【Top10 车辆（服务人次）】")
print(top10_bus)
print("=" * 70)

heatmap_data = np.array([
    top10_driver.values,
    top10_route.values,
    top10_stop.values,
    top10_bus.values
])

# 行/列标签
row_labels    = ['司机', '线路', '上车站点', '车辆']
col_labels    = [f'Top{i+1}' for i in range(10)]

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_data,
    annot=True,        # 标注数值
    fmt='g',           # 普通数字格式
    cmap='YlOrRd',     # 配色
    xticklabels=col_labels,
    yticklabels=row_labels
)

# 标题与样式
plt.title('公交服务能力 Top10 热力图对比', fontsize=16, pad=15)
plt.xlabel('排名', fontsize=12)
plt.ylabel('维度', fontsize=12)
plt.xticks(rotation=0)  # x轴标签不旋转

# 保存图片
save_path = os.path.join(desktop_path, "performance_heatmap.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
