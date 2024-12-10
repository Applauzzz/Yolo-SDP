# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # 父目录路径
# parent_dir = "/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/plot-new-save"

# # 存储所有实验数据
# experiments = {}

# # 遍历子目录
# for sub_dir in os.listdir(parent_dir):
#     sub_path = os.path.join(parent_dir, sub_dir)
#     # print(sub_path)
#     if os.path.isdir(sub_path):
#         # 找到 results.csv 文件
#         csv_file = os.path.join(sub_path, "results.csv")
#         exp_name = "lr="+sub_dir.split("-")[-1]+sub_dir.split("-")[0]+"-"+sub_dir.split("-")[1]  # 提取实验名
#         if os.path.exists(csv_file):
#             # 读取 CSV 文件
#             df = pd.read_csv(csv_file)
#             experiments[exp_name] = df

# # 按照实验名称排序
# sorted_experiments = dict(sorted(experiments.items()))

# # 绘制训练 loss
# plt.figure(figsize=(10, 6))
# for experiment_name, df in sorted_experiments.items():
#     df.columns = df.columns.str.strip() 
#     print(df.keys())
#     plt.plot(df['epoch'], df['train/loss'], label=f"{experiment_name} (Train Loss)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("training_loss_comparison.png")  # 保存图像
# plt.show()

# # 绘制训练 accuracy
# plt.figure(figsize=(10, 6))
# for experiment_name, df in sorted_experiments.items():
#     plt.plot(df['epoch'], df['metrics/accuracy_top1'], label=f"{experiment_name} (Accuracy Top-1)")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (Top-1)")
# plt.title("Training Accuracy Comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("training_accuracy_comparison.png")  # 保存图像
# plt.show()

# plt.figure(figsize=(10, 6))
# for experiment_name, df in sorted_experiments.items():
#     plt.plot(df['epoch'], df['val/loss'], label=f"{experiment_name} (Accuracy Top-5)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Validation Loss Comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("validation_loss_comparison.png")  # 保存图像
# plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
import random
# 父目录路径
parent_dir = "/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/plot-new-save"

# 存储所有实验数据
experiments = {}

best_acc = {}

# 遍历子目录
for sub_dir in os.listdir(parent_dir):
    sub_path = os.path.join(parent_dir, sub_dir)
    if os.path.isdir(sub_path):
        # 找到 results.csv 文件
        csv_file = os.path.join(sub_path, "results.csv")
        exp_name = "lr=" + sub_dir.split("-")[-1] + sub_dir.split("-")[0] + "-" + sub_dir.split("-")[1]  # 提取实验名
        if os.path.exists(csv_file):
            # 读取 CSV 文件
            df = pd.read_csv(csv_file)
            experiments[exp_name] = df

# 按照实验名称排序
sorted_experiments = dict(sorted(experiments.items()))


for experiment_name, df in sorted_experiments.items():
    df.columns = df.columns.str.strip()  # 去除列名的空格
    best_acc[experiment_name] = df['metrics/accuracy_top1'].max()
print(best_acc)

# 绘制训练 loss
plt.figure(figsize=(10, 6))
for experiment_name, df in sorted_experiments.items():
    df.columns = df.columns.str.strip()  # 去除列名的空格
    # 限制yolo-spd-0.05和yolo-spd-0.025的epoch最多到60
    # print(experiment_name)
    type = experiment_name.split("-")[-1]
    # print(type)
    # if "sdp" == type:
    #     # print(df["epoch"])
    #     df = df[df['epoch'] <= 60]
    plt.plot(df['epoch'], df['train/loss'], label=f"{experiment_name} (Train Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_comparison.png")  # 保存图像
plt.show()

# 绘制训练 accuracy
# plt.figure(figsize=(10, 6))
# for experiment_name, df in sorted_experiments.items():
#     # 限制yolo-spd-0.05和yolo-spd-0.025的epoch最多到60
#     type = experiment_name.split("-")[-1]
#     # print(type)
#     if "sdp" == type:
#         df = df[df['epoch'] <= 60+random.randint(-3, 5)]
#     plt.plot(df['epoch'], df['metrics/accuracy_top1'], label=f"{experiment_name} (Accuracy Top-1)")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (Top-1)")
# plt.title("Validation Accuracy Comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("Validation_accuracy_comparison.png")  # 保存图像
# plt.show()

# 绘制验证 loss
plt.figure(figsize=(10, 6))
for experiment_name, df in sorted_experiments.items():
    # 限制yolo-spd-0.05和yolo-spd-0.025的epoch最多到60
    # type = experiment_name.split("-")[-1]
    # # print(type)
    # if "sdp" == type:
    #     df = df[df['epoch'] <= 60]
    plt.plot(df['epoch'], df['val/loss'], label=f"{experiment_name} (Validation Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_loss_comparison.png")  # 保存图像
plt.show()
