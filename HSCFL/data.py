import os
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
import logging


class UserDataset(Dataset):
    """
    PyTorch Dataset wrapper for sparse matrices.

    Args:
        sparse_matrix: User-item interaction sparse matrix.
    """

    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix
        self.users = np.arange(sparse_matrix.shape[0])

    def __getitem__(self, index):
        user = self.users[index]
        return user, self.sparse_matrix[user].toarray().flatten()

    def __len__(self):
        return len(self.users)


def get_open_private_data(dataset_path):
    """
    Load interaction data for both open and private users from the specified directory.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        Tuple containing train/validation/test matrices for open and private users,
        counts, and mappings.
    """
    # Define file paths
    open_train_file = os.path.join(dataset_path, 'open_train.txt')
    open_test_file = os.path.join(dataset_path, 'open_test.txt')
    open_val_file = os.path.join(dataset_path, 'open_val.txt')

    private_train_file = os.path.join(dataset_path, 'private_train.txt')
    private_test_file = os.path.join(dataset_path, 'private_test.txt')
    private_val_file = os.path.join(dataset_path, 'private_val.txt')

    print(f"Loading data from path: {dataset_path}")

    def parse_file(file_path):
        """Parse user-item interactions from a text file."""
        user_to_items = {}
        if not os.path.exists(file_path):
            return user_to_items

        with open(file_path, 'r') as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                parts = line.strip().split()
                user_id = int(parts[0])
                item_ids = [int(item) for item in parts[1:] if item.strip()]
                user_to_items[user_id] = item_ids
        return user_to_items

    # Load open user data
    open_train_user_to_items = parse_file(open_train_file)
    open_test_user_to_items = parse_file(open_test_file)
    open_val_user_to_items = parse_file(open_val_file)

    # Load private user data
    private_train_user_to_items = parse_file(private_train_file)
    private_test_user_to_items = parse_file(private_test_file)
    private_val_user_to_items = parse_file(private_val_file)

    # Collect all unique items
    all_items = set()
    for user_dict in [open_train_user_to_items, open_test_user_to_items, open_val_user_to_items,
                      private_train_user_to_items, private_test_user_to_items, private_val_user_to_items]:
        for user, items in user_dict.items():
            all_items.update(items)

    # Collect unique users
    open_users = set()
    for user_dict in [open_train_user_to_items, open_test_user_to_items, open_val_user_to_items]:
        open_users.update(user_dict.keys())

    private_users = set()
    for user_dict in [private_train_user_to_items, private_test_user_to_items, private_val_user_to_items]:
        private_users.update(user_dict.keys())

    # Create mappings
    open_user_map = {old_id: new_id for new_id, old_id in enumerate(sorted(open_users))}
    private_user_map = {old_id: new_id for new_id, old_id in enumerate(sorted(private_users))}
    item_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_items))}

    n_open_users = len(open_user_map)
    n_private_users = len(private_user_map)
    n_items = len(item_map)

    print(f"Open users: {n_open_users}, Private users: {n_private_users}, Total items: {n_items}")

    def dict_to_sparse(user_to_items, n_users, n_items, user_map, item_map):
        """Convert interaction dictionary to a sparse CSR matrix."""
        rows = []
        cols = []

        for user, items in user_to_items.items():
            if user in user_map:
                u_idx = user_map[user]
                for item in items:
                    if item in item_map:
                        i_idx = item_map[item]
                        rows.append(u_idx)
                        cols.append(i_idx)

        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    # Create sparse matrices
    open_train_matrix = dict_to_sparse(open_train_user_to_items, n_open_users, n_items, open_user_map, item_map)
    open_test_matrix = dict_to_sparse(open_test_user_to_items, n_open_users, n_items, open_user_map, item_map)
    open_val_matrix = dict_to_sparse(open_val_user_to_items, n_open_users, n_items, open_user_map,
                                     item_map) if open_val_user_to_items else None

    private_train_matrix = dict_to_sparse(private_train_user_to_items, n_private_users, n_items, private_user_map,
                                          item_map)
    private_test_matrix = dict_to_sparse(private_test_user_to_items, n_private_users, n_items, private_user_map,
                                         item_map)
    private_val_matrix = dict_to_sparse(private_val_user_to_items, n_private_users, n_items, private_user_map,
                                        item_map) if private_val_user_to_items else None

    print(f"Open train matrix shape: {open_train_matrix.shape}, nnz: {open_train_matrix.nnz}")
    print(f"Private train matrix shape: {private_train_matrix.shape}, nnz: {private_train_matrix.nnz}")

    # Prepare datasets for training/validation/testing
    open_valid_in_data = open_train_matrix
    open_valid_out_data = open_val_matrix if open_val_matrix is not None else open_train_matrix
    open_test_in_data = open_train_matrix
    open_test_out_data = open_test_matrix

    private_valid_in_data = private_train_matrix
    private_valid_out_data = private_val_matrix if private_val_matrix is not None else private_train_matrix
    private_test_in_data = private_train_matrix
    private_test_out_data = private_test_matrix

    return (open_train_matrix, open_valid_in_data, open_valid_out_data, open_test_in_data, open_test_out_data,
            private_train_matrix, private_valid_in_data, private_valid_out_data, private_test_in_data,
            private_test_out_data,
            n_open_users, n_private_users, n_items, open_user_map, private_user_map, item_map)


def prepare_user_data_dict(train_matrix, user_map, user_type="users"):
    """
    Prepare a dictionary mapping user IDs to their sparse interaction vectors for efficient batching.

    Args:
        train_matrix: Sparse matrix containing training data.
        user_map: Dictionary mapping original user IDs to new indices.
        user_type: String description for logging.

    Returns:
        user_data_dict: Dictionary {user_id: sparse_vector}.
    """
    user_data_dict = {}

    for original_user_id, new_user_id in user_map.items():
        if new_user_id < train_matrix.shape[0]:
            user_data = train_matrix[new_user_id]
            if user_data.nnz > 0:  # Only keep users with interactions
                user_data_dict[new_user_id] = user_data

    print(f"Prepared {len(user_data_dict)} {user_type} with interactions for batch processing")
    return user_data_dict


def prepare_all_user_data_dict(open_train_matrix, private_train_matrix,
                               open_user_map, private_user_map, use_prefix=True):
    """
    Combine open and private user data into a single dictionary for unified training.
    """
    all_user_data = {}

    # Process open users
    for original_user_id, new_user_id in open_user_map.items():
        if new_user_id < open_train_matrix.shape[0]:
            user_data = open_train_matrix[new_user_id]
            if user_data.nnz > 0:
                if use_prefix:
                    client_id = f"open_{original_user_id}"
                else:
                    client_id = new_user_id
                all_user_data[client_id] = user_data

    # Process private users
    for original_user_id, new_user_id in private_user_map.items():
        if new_user_id < private_train_matrix.shape[0]:
            user_data = private_train_matrix[new_user_id]
            if user_data.nnz > 0:
                if use_prefix:
                    client_id = f"private_{original_user_id}"
                else:
                    client_id = new_user_id + len(open_user_map)
                all_user_data[client_id] = user_data

    print(f"Prepared {len(all_user_data)} total users for unified training")
    return all_user_data


def prepare_sparse_batch(batch_users, user_data_dict, n_items, device):
    """
    Construct a dense tensor batch from sparse user data efficiently.

    Args:
        batch_users: List of user IDs for the current batch.
        user_data_dict: Dictionary containing sparse user vectors.
        n_items: Total number of items.
        device: Computation device.

    Returns:
        batch_tensor: Dense tensor of shape [batch_size, n_items].
    """
    batch_data = []

    for user_id in batch_users:
        if user_id in user_data_dict:
            user_data = user_data_dict[user_id]
            if user_data.nnz > 0:
                user_dense = np.zeros(n_items, dtype=np.float32)
                user_dense[user_data.indices] = user_data.data
                batch_data.append(user_dense)

    if batch_data:
        return torch.FloatTensor(np.vstack(batch_data)).to(device)
    else:
        return torch.zeros(len(batch_users), n_items, dtype=torch.float32, device=device)


def merge_validation_data(open_valid_in_data, open_valid_out_data,
                          private_valid_in_data, private_valid_out_data):
    """
    Merge validation datasets from open and private users.
    """
    merged_valid_in_data = sparse.vstack([open_valid_in_data, private_valid_in_data])
    merged_valid_out_data = sparse.vstack([open_valid_out_data, private_valid_out_data])

    print(f"Merged validation data shape: {merged_valid_in_data.shape}")
    return merged_valid_in_data, merged_valid_out_data


def merge_test_data(open_test_in_data, open_test_out_data,
                    private_test_in_data, private_test_out_data):
    """
    Merge test datasets from open and private users.
    """
    merged_test_in_data = sparse.vstack([open_test_in_data, private_test_in_data])
    merged_test_out_data = sparse.vstack([open_test_out_data, private_test_out_data])

    print(f"Merged test data shape: {merged_test_in_data.shape}")
    return merged_test_in_data, merged_test_out_data