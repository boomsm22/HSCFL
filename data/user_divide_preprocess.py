import os
import numpy as np
import pandas as pd
import random
import sys
import shutil
import argparse
from collections import defaultdict


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def calculate_user_activity(dataset_path):
    """
    Calculate user activity based on the number of interactions

    Args:
        dataset_path: Path to dataset directory

    Returns:
        dict: Dictionary with user activity scores
    """
    user_activity = defaultdict(int)

    train_file = os.path.join(dataset_path, 'train.txt')
    with open(train_file, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip().split(' ')
                user = int(line[0])
                items = [int(i) for i in line[1:] if i]
                user_activity[user] += len(items)

    val_file = os.path.join(dataset_path, 'val.txt')
    if os.path.exists(val_file):
        with open(val_file, 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip().split(' ')
                    user = int(line[0])
                    items = [int(i) for i in line[1:] if i]
                    user_activity[user] += len(items)

    test_file = os.path.join(dataset_path, 'test.txt')
    with open(test_file, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip().split(' ')
                user = int(line[0])
                items = [int(i) for i in line[1:] if i]
                user_activity[user] += len(items)

    return user_activity


def divide_users_by_activity(user_activity, seed=2022):
    """
    Divide users into open and private based on their activity

    Args:
        user_activity: Dictionary with user activity scores
        seed: Random seed for reproducibility

    Returns:
        set, set: Sets of open and private users
    """
    set_seed(seed)

    sorted_users = sorted(user_activity.keys(), key=lambda u: user_activity[u], reverse=True)

    mid_point = len(sorted_users) // 2
    high_activity_users = sorted_users[:mid_point]
    low_activity_users = sorted_users[mid_point:]

    open_users = set()
    private_users = set()

    for user in high_activity_users:
        if random.random() < 0.8:  # 80% probability
            open_users.add(user)
        else:
            private_users.add(user)

    for user in low_activity_users:
        if random.random() < 0.2:
            open_users.add(user)
        else:
            private_users.add(user)

    return open_users, private_users


def divide_users_randomly(user_activity, open_rate=0.5, seed=2022):
    """
    Randomly divide users into open and private users

    Args:
        user_activity: Dictionary with user activity scores
        open_rate: Proportion of open users (between 0-1)
        seed: Random seed for reproducibility

    Returns:
        set, set: Sets of open and private users
    """
    set_seed(seed)

    all_users = list(user_activity.keys())

    random.shuffle(all_users)

    num_open_users = int(len(all_users) * open_rate)

    open_users = set(all_users[:num_open_users])
    private_users = set(all_users[num_open_users:])

    return open_users, private_users


def process_dataset(dataset_name, method='activity', open_rate=0.5, seed=2022):
    """
    Process dataset to create open and private user splits

    Args:
        dataset_name: Name of the dataset (gowalla, yelp2018, amazon-book)
        method: Split method ('activity' or 'random')
        open_rate: Proportion of open users (only used in random method)
        seed: Random seed for reproducibility
    """
    print(f"Processing dataset: {dataset_name}")
    print(f"Split method: {'Activity-based' if method == 'activity' else 'Random split'}")
    if method == 'random':
        print(f"Open user rate: {open_rate * 100}%")

    data_dir = '.'
    dataset_path = os.path.join(data_dir, dataset_name)

    processed_dir = os.path.join(data_dir, 'processed_data')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    dataset_processed_dir = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(dataset_processed_dir):
        os.makedirs(dataset_processed_dir)

    if method == 'activity':
        method_dir = os.path.join(dataset_processed_dir, 'activity_based')
    else:
        rate_str = f"{int(open_rate * 100)}"
        method_dir = os.path.join(dataset_processed_dir, rate_str)

    if not os.path.exists(method_dir):
        os.makedirs(method_dir)

    user_activity = calculate_user_activity(dataset_path)
    print(f"Total users in {dataset_name}: {len(user_activity)}")

    if method == 'activity':
        open_users, private_users = divide_users_by_activity(user_activity, seed)
    else:
        open_users, private_users = divide_users_randomly(user_activity, open_rate, seed)

    print(f"Open users: {len(open_users)} ({len(open_users) / len(user_activity) * 100:.2f}%)")
    print(f"Private users: {len(private_users)} ({len(private_users) / len(user_activity) * 100:.2f}%)")

    with open(os.path.join(method_dir, 'open_users.txt'), 'w') as f:
        for user in sorted(open_users):
            f.write(f"{user}\n")

    with open(os.path.join(method_dir, 'private_users.txt'), 'w') as f:
        for user in sorted(private_users):
            f.write(f"{user}\n")

    for file_type in ['train.txt', 'val.txt', 'test.txt']:
        if not os.path.exists(os.path.join(dataset_path, file_type)):
            continue

        open_lines = []
        private_lines = []

        with open(os.path.join(dataset_path, file_type), 'r') as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip()
                    user = int(line.split(' ')[0])

                    if user in open_users:
                        open_lines.append(line)
                    elif user in private_users:
                        private_lines.append(line)

        with open(os.path.join(method_dir, f'open_{file_type}'), 'w') as f:
            for line in open_lines:
                f.write(f"{line}\n")

        with open(os.path.join(method_dir, f'private_{file_type}'), 'w') as f:
            for line in private_lines:
                f.write(f"{line}\n")

        shutil.copy2(os.path.join(dataset_path, file_type), os.path.join(method_dir, file_type))

        print(f"Processed {file_type}: {len(open_lines)} open users, {len(private_lines)} private users")

    print(f"Dataset {dataset_name} processed successfully. Files saved to {method_dir}\n")

    return open_users, private_users, user_activity


def analyze_activity_distribution(dataset_name, open_users, private_users, user_activity, method):
    """
    Analyze and print activity distribution between open and private users

    Args:
        dataset_name: Name of the dataset
        open_users: Set of open users
        private_users: Set of private users
        user_activity: Dictionary with user activity scores
        method: Split method description
    """
    open_activity = [user_activity[u] for u in open_users]
    private_activity = [user_activity[u] for u in private_users]

    print(f"\nActivity Analysis for {dataset_name} (Method: {method}):")
    print(f"Average interactions per open user: {sum(open_activity) / len(open_activity):.2f}")
    print(f"Average interactions per private user: {sum(private_activity) / len(private_activity):.2f}")
    print(f"Max interactions for open users: {max(open_activity)}")
    print(f"Max interactions for private users: {max(private_activity)}")
    print(f"Min interactions for open users: {min(open_activity)}")
    print(f"Min interactions for private users: {min(private_activity)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process datasets for open/private user splits')

    parser.add_argument('--datasets', nargs='+',
                        default=['gowalla', 'yelp2018', 'amazon'],
                        help='Dataset names to process (default: gowalla yelp2018 amazon)')

    parser.add_argument('--method', choices=['activity', 'random'],
                        default='activity',
                        help='Split method: activity-based or random (default: activity)')

    parser.add_argument('--rate', type=float, default=0.5,
                        help='Open user rate for random split (default: 0.5)')

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed (default: 2022)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.method == 'random' and not (0 < args.rate < 1):
        print("Error: Open user rate must be between 0 and 1")
        sys.exit(1)

    processed_dir = os.path.join('.', 'processed_data')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print(f"Configuration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Method: {args.method}")
    if args.method == 'random':
        print(f"  Open rate: {args.rate}")
    print(f"  Seed: {args.seed}")
    print("=" * 50)

    for dataset in args.datasets:
        dataset_path = os.path.join('.', dataset)

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory {dataset_path} not found. Skipping.")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 50}")

        if args.method == 'activity':
            open_users, private_users, user_activity = process_dataset(
                dataset, method='activity', seed=args.seed
            )
            method_name = "Activity-based"
        else:
            open_users, private_users, user_activity = process_dataset(
                dataset, method='random', open_rate=args.rate, seed=args.seed
            )
            method_name = f"Random split ({int(args.rate * 100)}%)"

        analyze_activity_distribution(dataset, open_users, private_users, user_activity, method_name)