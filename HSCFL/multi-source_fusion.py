import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
import copy
import random
import gc
from datetime import datetime
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

from utils import set_seed, init_logging, compute_metrics_gpu
from data import get_open_private_data


class FusionNetwork(nn.Module):
    """
    Fusion Network: MLP-based multi-feature fusion model.
    Input features: [score_z, prob_z, score_g, uncertainty, popularity, activity]
    """

    def __init__(self, input_dim=6, mlp_dims=[32, 16], dropout_p=0.1):
        super(FusionNetwork, self).__init__()

        self.input_dim = input_dim
        self.mlp_dims = mlp_dims

        layers = []
        prev_dim = input_dim

        for dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
        logging.info(f"FusionNetwork initialized: input_dim={input_dim}, mlp_dims={mlp_dims}")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features):
        return self.mlp(features)


class ScoreLoader:
    """
    Efficiently loads ALL precomputed scores and features (including uncertainty) using memory mapping.
    """

    def __init__(self, cache_dir, n_users, n_items):
        self.cache_dir = cache_dir
        self.n_users = n_users
        self.n_items = n_items

        required_files = [
            'base_scores.dat', 'base_probs.dat', 'lightgcn_scores.dat',
            'uncertainty.dat', 'popularity.npy', 'activity.npy'
        ]

        for filename in required_files:
            if not os.path.exists(os.path.join(cache_dir, filename)):
                raise FileNotFoundError(f"Required file not found: {filename}")

        # Load Memmaps
        self.base_scores = np.memmap(os.path.join(cache_dir, 'base_scores.dat'), dtype='float32', mode='r',
                                     shape=(n_users, n_items))
        self.base_probs = np.memmap(os.path.join(cache_dir, 'base_probs.dat'), dtype='float32', mode='r',
                                    shape=(n_users, n_items))
        self.lightgcn_scores = np.memmap(os.path.join(cache_dir, 'lightgcn_scores.dat'), dtype='float32', mode='r',
                                         shape=(n_users, n_items))
        self.uncertainty = np.memmap(os.path.join(cache_dir, 'uncertainty.dat'), dtype='float32', mode='r',
                                     shape=(n_users, n_items))

        # Load Static Arrays
        self.popularity = np.load(os.path.join(cache_dir, 'popularity.npy'))
        self.activity = np.load(os.path.join(cache_dir, 'activity.npy'))

        logging.info(f"ScoreLoader initialized: [{n_users} x {n_items}]. All features loaded.")

    def get_user_item_features(self, user_ids, item_ids):
        """
        Construct feature vectors for given user-item pairs.
        Uses vectorized NumPy indexing for speed.
        """
        batch_size = len(user_ids)
        features = np.zeros((batch_size, 6), dtype=np.float32)

        # Feature 1: Personalized Logit
        features[:, 0] = self.base_scores[user_ids, item_ids]
        # Feature 2: Personalized Probability
        features[:, 1] = self.base_probs[user_ids, item_ids]
        # Feature 3: Collaborative Logit
        features[:, 2] = self.lightgcn_scores[user_ids, item_ids]
        # Feature 4: Uncertainty (Now loaded directly here)
        features[:, 3] = self.uncertainty[user_ids, item_ids]
        # Feature 5: Item Popularity
        features[:, 4] = self.popularity[item_ids]
        # Feature 6: User Activity
        features[:, 5] = self.activity[user_ids]

        return features


def prepare_training_triplets(open_train_matrix):
    """
    Prepare epoch-level training triplets (user, pos, neg) with 1:1 ratio.
    """
    user_item_coo = open_train_matrix.tocoo()
    n_users, n_items = open_train_matrix.shape
    user_ids = user_item_coo.row
    pos_item_ids = user_item_coo.col

    user_items_dict = {}
    for u, i in zip(user_ids, pos_item_ids):
        if u not in user_items_dict: user_items_dict[u] = set()
        user_items_dict[u].add(i)

    neg_item_ids = []
    for user_id in user_ids:
        user_pos_items = user_items_dict[user_id]
        neg_item = random.randint(0, n_items - 1)
        while neg_item in user_pos_items:
            neg_item = random.randint(0, n_items - 1)
        neg_item_ids.append(neg_item)
    return user_ids, pos_item_ids, np.array(neg_item_ids)


def evaluate_fusion(model, valid_in_data, valid_out_data, score_loader,
                    scaler, device, batch_size=500, ndcg20_only=False):
    """
    Custom evaluation loop for Fusion Model using GPU acceleration.
    """
    model.eval()
    n_users = valid_in_data.shape[0]
    n_items = valid_in_data.shape[1]

    # 1. Determine metrics
    if ndcg20_only:
        k_list = [20]
    else:
        k_list = [10, 20, 50]

    all_metric_values = {}

    with torch.no_grad():
        num_batches = (n_users + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_users)
            batch_user_ids = np.arange(start_idx, end_idx)
            batch_size_curr = len(batch_user_ids)

            batch_predictions = torch.full((batch_size_curr, n_items), -float('inf'), device=device)

            item_chunk_size = 2000

            for i_start in range(0, n_items, item_chunk_size):
                i_end = min(i_start + item_chunk_size, n_items)
                curr_items_len = i_end - i_start

                u_ids_exp = np.repeat(batch_user_ids, curr_items_len)
                i_ids_exp = np.tile(np.arange(i_start, i_end), batch_size_curr)

                features = score_loader.get_user_item_features(u_ids_exp, i_ids_exp)

                feat_scaled = scaler.transform(features)
                feat_tensor = torch.FloatTensor(feat_scaled).to(device)
                scores = model(feat_tensor).squeeze()
                batch_predictions[:, i_start:i_end] = scores.reshape(batch_size_curr, curr_items_len)
                del feat_tensor, scores

            if valid_in_data is not valid_out_data:
                batch_data_in = valid_in_data[start_idx:end_idx]
                batch_data_in_dense = torch.FloatTensor(batch_data_in.toarray()).to(device)
                batch_predictions[batch_data_in_dense > 0] = -float('inf')

            batch_data_out = valid_out_data[start_idx:end_idx]
            batch_gt_dense = torch.FloatTensor(batch_data_out.toarray()).to(device)

            batch_results = compute_metrics_gpu(batch_predictions, batch_gt_dense, k_list)

            for name, values in batch_results.items():
                if name not in all_metric_values:
                    all_metric_values[name] = []
                all_metric_values[name].extend(values.cpu().numpy())

            del batch_predictions, batch_gt_dense

    if ndcg20_only:
        return np.mean(all_metric_values['ndcg@20']) if 'ndcg@20' in all_metric_values else 0.0
    else:
        results = {}
        for name, values in all_metric_values.items():
            results[name] = np.mean(values) if len(values) > 0 else 0.0
        return results


def train_fusion_network(model, optimizer, open_train_matrix,
                         open_valid_in_data, open_valid_out_data,
                         score_loader, scaler,
                         device, n_epochs, batch_size=1024,
                         early_stopping=20, eval_every=5):
    logging.info("Starting fusion network training...")
    best_ndcg = -np.inf
    best_epoch = 0
    stopping_step = 0
    best_state = None
    best_val_metrics = {}

    logging.info("Fitting feature scaler...")
    u_ids, p_ids, n_ids = prepare_training_triplets(open_train_matrix)

    p_feats = score_loader.get_user_item_features(u_ids, p_ids)
    n_feats = score_loader.get_user_item_features(u_ids, n_ids)

    scaler.fit(np.vstack([p_feats, n_feats]))
    del u_ids, p_ids, n_ids, p_feats, n_feats
    gc.collect()

    for epoch in range(n_epochs):
        model.train()
        user_ids, pos_ids, neg_ids = prepare_training_triplets(open_train_matrix)
        n_samples = len(user_ids)
        n_batches = (n_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        indices = np.random.permutation(n_samples)
        user_ids = user_ids[indices]
        pos_ids = pos_ids[indices]
        neg_ids = neg_ids[indices]

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)

            batch_u = user_ids[start:end]
            batch_p = pos_ids[start:end]
            batch_n = neg_ids[start:end]

            pos_feats = score_loader.get_user_item_features(batch_u, batch_p)
            neg_feats = score_loader.get_user_item_features(batch_u, batch_n)

            pos_tensor = torch.FloatTensor(scaler.transform(pos_feats)).to(device)
            neg_tensor = torch.FloatTensor(scaler.transform(neg_feats)).to(device)

            optimizer.zero_grad()

            pos_scores = model(pos_tensor).squeeze()
            neg_scores = model(neg_tensor).squeeze()

            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            current_ndcg = evaluate_fusion(
                model, open_valid_in_data, open_valid_out_data,
                score_loader, scaler, device, batch_size=500, ndcg20_only=True
            )

            logging.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val NDCG@20={current_ndcg:.10f}")

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                stopping_step = 0
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = {'ndcg@20': best_ndcg}
            else:
                stopping_step += eval_every

            if stopping_step >= early_stopping: break

    if best_state:
        model.load_state_dict(best_state)
        logging.info(f"Training finished. Computing full metrics for best model (Epoch {best_epoch})...")
        best_val_metrics = evaluate_fusion(
            model, open_valid_in_data, open_valid_out_data,
            score_loader, scaler, device, batch_size=500, ndcg20_only=False
        )

    return model, best_val_metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-source Fusion ")
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--data_path', type=str, default='../data/processed_data')
    parser.add_argument('--ratio', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mlp_dims', nargs='+', type=int, default=[32, 16])
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2024)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    timestamp = int(datetime.now().timestamp())
    result_dir = f"fusion_results/{args.dataset}_{args.ratio}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    init_logging(os.path.join(result_dir, 'fusion_training.log'))
    logging.info(f"Configuration: {vars(args)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(args.data_path, args.dataset, str(args.ratio))
    data = get_open_private_data(dataset_path)
    open_train, open_val_in, open_val_out, open_test_in, open_test_out = data[0:5]
    n_users, n_items = data[10], data[12]

    cache_dir = f"precompute_cache/{args.dataset}/{args.ratio}"
    if not os.path.exists(cache_dir): raise FileNotFoundError(f"Cache not found: {cache_dir}")

    score_loader = ScoreLoader(cache_dir, n_users, n_items)
    scaler = StandardScaler()

    model = FusionNetwork(input_dim=6, mlp_dims=args.mlp_dims, dropout_p=args.dropout_p).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, best_val_metrics = train_fusion_network(
        model, optimizer, open_train, open_val_in, open_val_out,
        score_loader, scaler, device,
        args.n_epochs, args.batch_size, args.early_stopping, args.eval_every
    )

    logging.info("=== Best Validation Metrics ===")
    for k, v in best_val_metrics.items(): logging.info(f"  {k}: {v:.10f}")

    logging.info("=== Final Evaluation ===")
    test_results = evaluate_fusion(
        model, open_test_in, open_test_out, score_loader, scaler, device, args.eval_batch_size
    )
    for k, v in test_results.items(): logging.info(f"  {k}: {v:.10f}")

    logging.info("Multi-feature fusion completed successfully!")


if __name__ == "__main__":
    main()