import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from SDSE_Net import SDSE_Net
from trainer import trainer_synapse, trainer_acdc, trainer_busi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="./data/Synapse/train_npz",
    help="root dir for data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
)
parser.add_argument(
    "--num_classes", type=int, default=9, help="output channel of network"
)
parser.add_argument(
    "--max_iterations", type=int, default=30000, help="maximum epoch number to train"
)
parser.add_argument(
    "--max_epochs", type=int, default=30, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--gpu", type=str, default="0", help="ID of GPU to use")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.001, help="segmentation network learning rate"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
# 想要修改种子只需要修改seed的默认值即可，或者在命令行中指定--seed参数
# parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# parser.add_argument("--seed", type=int, default=2026, help="random seed")
parser.add_argument("--pretrained_path", type=str, default="net_B.pth")
args = parser.parse_args()


def seed_torch(seed):
    torch.manual_seed(seed)  # 固定CPU上Torch的随机种子
    torch.cuda.manual_seed(seed)  # 固定单个GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 固定所有GPU的随机种子
    np.random.seed(seed)  # 固定NumPy的随机种子
    random.seed(seed)  # 固定Python原生random库的随机种子
    torch.backends.cudnn.deterministic = (
        True  # 强制CUDNN使用确定性算法（而非加速的非确定性算法）
    )
    torch.backends.cudnn.benchmark = (
        False  # 关闭CUDNN的基准测试（避免自动选不同卷积算法带来的随机性）
    )
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # 固定Python哈希种子（避免字典/集合等哈希随机）
    torch.use_deterministic_algorithms(
        True, warn_only=True
    )  # 强制Torch所有算法使用确定性实现
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # 固定CUDA矩阵运算的内存配置（避免CUBLAS的非确定性）
    )


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_torch(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "ACDC": {
            #'Dataset': ACDC_dataset,  # datasets.dataset_acdc.BaseDataSets,
            "root_path": "./data/ACDC",
            "list_dir": None,
            "num_classes": 4,
            "worker_init_fn": worker_init_fn,
        },
        "Synapse": {
            "root_path": "./data/Synapse/train_npz",
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 9,
            "worker_init_fn": worker_init_fn,
        },
        "busi": {
            "root_path": "./data/busi",  # base_dir
            "list_dir": "./data/busi",  # train.txt 和 val.txt 所在目录
            "num_classes": 2,
            "worker_init_fn": worker_init_fn,
        },
    }
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.is_pretrain = True
    args.exp = "rwkv" + dataset_name + str(args.img_size)
    snapshot_path = "./outputs/{}/{}".format(args.exp, "exp1")
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path = (
        snapshot_path + "_" + str(args.max_iterations)[0:2] + "k"
        if args.max_iterations != 30000
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_epo" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = (
        snapshot_path + "_lr" + str(args.base_lr)
        if args.base_lr != 0.01
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    net = SDSE_Net(
        in_channels=1,
        img_size=args.img_size,
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path,
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)

    net = net.to("cuda")
    trainer = {
        "Synapse": trainer_synapse,
        "ACDC": trainer_acdc,
        "busi": trainer_busi,
    }
    trainer[dataset_name](args, net, snapshot_path, worker_init_fn)
