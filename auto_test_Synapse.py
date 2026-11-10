import os
import subprocess
import argparse
import re
from tqdm import tqdm


def find_and_sort_weights(weights_dir):
    """
    Find all .pth files in the directory and sort them by epoch number.
    Epoch number is extracted from the filename.
    """
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]

    weights_with_epochs = []
    for weight_file in weight_files:
        # Try to extract number from filename, e.g., 'epoch_40.pth' -> 40
        match = re.search(r"\d+", weight_file)
        if match:
            epoch_num = int(match.group())
            full_path = os.path.join(weights_dir, weight_file)
            weights_with_epochs.append((epoch_num, full_path))
        else:
            print(
                f"Warning: Could not extract epoch number from '{weight_file}'. It will be skipped."
            )

    # Sort by epoch number (the first element of the tuple)
    weights_with_epochs.sort()

    # Return only the sorted full paths
    return [path for epoch, path in weights_with_epochs]


def run_test(weight_path):
    """
    Constructs and runs the test command for a single weight file.
    """
    print(f"\n{'='*20}")
    print(f"Testing weight: {os.path.basename(weight_path)}")
    print(f"{'='*20}")

    command = [
        "python",
        "test.py",
        "--dataset",
        "Synapse",
        "--path_specific",
        weight_path,
    ]

    try:
        # Using subprocess.run to execute the command
        # We pass check=True to raise an exception if the command fails
        subprocess.run(command, check=True)
        print(f"\nSuccessfully finished testing {os.path.basename(weight_path)}")
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while testing {os.path.basename(weight_path)}.")
        print(f"Error: {e}")
    except FileNotFoundError:
        print(
            "\nError: 'python' command not found. Make sure your environment is activated."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Automate testing for multiple RWKV-UNet weights."
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Path to the directory containing the .pth weight files.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.weights_dir):
        print(f"Error: Directory not found at '{args.weights_dir}'")
        return

    sorted_weights = find_and_sort_weights(args.weights_dir)

    if not sorted_weights:
        print(f"No '.pth' files with epoch numbers found in '{args.weights_dir}'.")
        return

    print(f"Found and sorted {len(sorted_weights)} weight files to test.")

    for weight_path in tqdm(sorted_weights, desc="Overall Test Progress"):
        run_test(weight_path)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
