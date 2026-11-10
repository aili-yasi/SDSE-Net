import os
import subprocess
import argparse

def run_workflow():
    # --- 1. 定义训练参数 ---
    # 您可以在这里修改所有需要的训练参数
    # 这些参数将用于构建训练命令和定位输出文件夹
    parser = argparse.ArgumentParser(description="Run a training and testing workflow for Synapse dataset.")
    
    # 从train.py复制过来的参数，并设置默认值
    parser.add_argument("--dataset", type=str, default="Synapse", help="Dataset name")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size per GPU")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--pretrained_path", type=str, default="net_B.pth", help="Path to the pretrained weights")

    # 想要修改种子只需要修改seed的默认值即可，或者在命令行中指定--seed参数
    # parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    # parser.add_argument("--seed", type=int, default=2026, help="random seed")

    parser.add_argument("--gpu", type=str, default="0", help="ID of GPU to use")
    parser.add_argument("--max_iterations", type=int, default=30000, help="Maximum iterations")
    
    args = parser.parse_args()

    # --- 2. 执行训练 ---
    print("="*30)
    print("STEP 1: Starting training process...")
    print("="*30)
    
    train_command = [
        "python", "train.py",
        "--dataset", args.dataset,
        "--max_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--base_lr", str(args.base_lr),
        "--img_size", str(args.img_size),
        "--seed", str(args.seed),
        "--gpu", args.gpu,
        "--pretrained_path", args.pretrained_path,
    ]
    
    try:
        subprocess.run(train_command, check=True)
        print("\nTraining process completed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError during training process: {e}")
        return # 如果训练失败，则终止脚本

    # --- 3. 构建权重目录路径 ---
    # 这部分逻辑完全复制自 train.py，以确保路径匹配
    print("\n" + "="*30)
    print("STEP 2: Calculating weights directory path...")
    print("="*30)
    
    exp_name = "rwkv" + args.dataset + str(args.img_size)
    snapshot_path = f"./outputs/{exp_name}/exp1"
    snapshot_path += "_pretrain" # is_pretrain is always True for this workflow
    if args.max_iterations != 30000:
        snapshot_path += f"_{str(args.max_iterations)[0:2]}k"
    if args.max_epochs != 30:
        snapshot_path += f"_epo{args.max_epochs}"
    snapshot_path += f"_bs{args.batch_size}"
    if args.base_lr != 0.01:
        snapshot_path += f"_lr{args.base_lr}"
    snapshot_path += f"_{args.img_size}"
    if args.seed != 1234:
        snapshot_path += f"_s{args.seed}"

    print(f"Calculated weights directory: {snapshot_path}")

    if not os.path.isdir(snapshot_path):
        print(f"Error: The calculated directory '{snapshot_path}' does not exist. Please check the path logic.")
        return

    # --- 4. 执行自动化测试 ---
    print("\n" + "="*30)
    print("STEP 3: Starting automated testing process...")
    print("="*30)

    test_command = [
        "python", "auto_test_Synapse.py",
        "--weights_dir", snapshot_path
    ]

    try:
        subprocess.run(test_command, check=True)
        print("\nAutomated testing completed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError during automated testing process: {e}")
        return

    print("\nWorkflow finished!")

if __name__ == "__main__":
    run_workflow()
