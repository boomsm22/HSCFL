import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
import random
import gc
import copy
from torch.optim import Adam
from scipy import sparse
from datetime import datetime

from model import MultiVAE
from hybrid_finetuning import GatedLoRAMultiVAE
from utils import set_seed, init_logging, compute_metrics_gpu
from data import get_open_private_data


class DistillLightGCN(nn.Module):
    """
    LightGCN with listwise knowledge distillation support.
    """

    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, reg_weight=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.adjacency_matrix = None

    def create_adjacency_matrix(self, user_item_matrix):
        """Create normalized adjacency matrix for the bipartite graph."""
        coo = user_item_matrix.tocoo()
        row = np.concatenate([coo.row, coo.col + self.n_users])
        col = np.concatenate([coo.col + self.n_users, coo.row])
        data = np.ones(len(row))
        adj = sparse.coo_matrix((data, (row, col)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        rowsum = np.array(adj.sum(1))

        # Handle divide by zero warning safely
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def set_adjacency_matrix(self, adj):
        self.adjacency_matrix = adj

    def computer(self):
        """Propagate embeddings through the graph."""
        device = self.user_embedding.weight.device
        adj_indices = torch.from_numpy(np.vstack([self.adjacency_matrix.row, self.adjacency_matrix.col])).long().to(
            device)
        adj_values = torch.from_numpy(self.adjacency_matrix.data).float().to(device)
        adj = torch.sparse_coo_tensor(adj_indices, adj_values, self.adjacency_matrix.shape, device=device)
        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [ego_emb]
        for _ in range(self.n_layers):
            ego_emb = torch.sparse.mm(adj, ego_emb)
            embs.append(ego_emb)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]

    def getUsersRating(self, users):
        """Compute scores for specific users against all items."""
        u_emb, i_emb = self.computer()
        return torch.matmul(u_emb[users], i_emb.t())

    def compute_listwise_scores(self, user_ids, item_lists, user_all_emb=None, item_all_emb=None):
        """Compute scores for specific user-item pairs efficiently."""
        if user_all_emb is None or item_all_emb is None:
            user_all_emb, item_all_emb = self.computer()

        batch_u = user_all_emb[user_ids]
        batch_i = item_all_emb[item_lists]

        return torch.sum(batch_u.unsqueeze(1) * batch_i, dim=2)


class TeacherScoreLoader:
    """Loads precomputed teacher scores for distillation."""

    def __init__(self, cache_dir, n_users, n_items):
        path = os.path.join(cache_dir, 'base_scores.dat')
        self.base_scores = np.memmap(path, dtype='float32', mode='r', shape=(n_users, n_items))

    def get_listwise_teacher_scores(self, user_ids, item_lists):
        """
        Vectorized implementation using Advanced Indexing.
        """
        scores = self.base_scores[user_ids[:, None], item_lists]
        return scores


def load_base_model_for_precompute(lora_path, p_dims, lora_ranks, gate_dims, lora_dropout, device):
    """
    Load the Gated LoRA model from the saved checkpoint.
    """
    logging.info("Loading Gated LoRA Model for precomputation...")

    dummy_pretrained = MultiVAE(p_dims=p_dims, dropout_p=0.5)

    base_model = GatedLoRAMultiVAE(dummy_pretrained, lora_ranks, gate_dims, lora_dropout).to(device)

    logging.info(f"Loading weights from: {lora_path}")
    base_model.load_state_dict(torch.load(lora_path, map_location=device), strict=True)

    base_model.eval()
    for param in base_model.parameters(): param.requires_grad = False

    return base_model


def precompute_base_scores(base_model, open_train, device, batch_size, save_dir):
    """Compute and save Base Model scores."""
    n_users, n_items = open_train.shape
    os.makedirs(save_dir, exist_ok=True)
    scores = np.memmap(os.path.join(save_dir, 'base_scores.dat'), dtype='float32', mode='w+', shape=(n_users, n_items))
    with torch.no_grad():
        num_batches = (n_users + batch_size - 1) // batch_size
        for idx in range(num_batches):
            start, end = idx * batch_size, min((idx + 1) * batch_size, n_users)
            batch = torch.FloatTensor(open_train[start:end].toarray()).to(device)
            recon, _, _ = base_model(batch)
            scores[start:end] = recon.cpu().numpy()
            scores.flush()
    np.save(os.path.join(save_dir, 'base_scores_shape.npy'), {'n_users': n_users, 'n_items': n_items})


def precompute_base_probs_from_scores(cache_dir, n_users, n_items, batch_size=100):
    """Convert scores to probabilities via Softmax."""
    scores_path = os.path.join(cache_dir, 'base_scores.dat')
    probs_path = os.path.join(cache_dir, 'base_probs.dat')
    if not os.path.exists(scores_path): raise FileNotFoundError(f"BaseModel scores not found: {scores_path}")
    scores = np.memmap(scores_path, dtype='float32', mode='r', shape=(n_users, n_items))
    probs = np.memmap(probs_path, dtype='float32', mode='w+', shape=(n_users, n_items))
    for idx in range((n_users + batch_size - 1) // batch_size):
        start, end = idx * batch_size, min((idx + 1) * batch_size, n_users)
        batch_scores = torch.FloatTensor(np.array(scores[start:end]))
        probs[start:end] = F.softmax(batch_scores, dim=1).numpy()
        probs.flush()


def precompute_static_features(train_matrix, save_dir):
    """Compute popularity and activity features."""
    os.makedirs(save_dir, exist_ok=True)
    pop = np.array(train_matrix.sum(axis=0)).flatten().astype('float32')
    act = np.array(train_matrix.sum(axis=1)).flatten().astype('float32')
    np.save(os.path.join(save_dir, 'popularity.npy'), pop)
    np.save(os.path.join(save_dir, 'activity.npy'), act)


def precompute_uncertainty_table(cache_dir, n_users, n_items, batch_size=100):
    """Compute uncertainty based on probability distribution."""
    probs = np.memmap(os.path.join(cache_dir, 'base_probs.dat'), dtype='float32', mode='r', shape=(n_users, n_items))
    unc = np.memmap(os.path.join(cache_dir, 'uncertainty.dat'), dtype='float32', mode='w+', shape=(n_users, n_items))
    for idx in range((n_users + batch_size - 1) // batch_size):
        start, end = idx * batch_size, min((idx + 1) * batch_size, n_users)
        batch_p = probs[start:end]
        batch_unc = np.zeros_like(batch_p)
        for i, p in enumerate(batch_p):
            mu, sigma = np.mean(p), np.std(p)
            sigma = max(sigma, 1e-8)
            batch_unc[i] = np.exp(-((p - mu) ** 2) / (2 * sigma ** 2))
        unc[start:end] = batch_unc
        unc.flush()


def prepare_listwise_training_data(matrix, K=4):
    """Prepare training triplets (1 positive + K negatives)."""
    coo = matrix.tocoo()
    users, pos_items = coo.row, coo.col
    user_items = {}
    for u, i in zip(users, pos_items):
        if u not in user_items: user_items[u] = set()
        user_items[u].add(i)
    all_users, all_lists = [], []
    for i, (u, pos) in enumerate(zip(users, pos_items)):
        lst = [pos]
        count = 0
        while count < K:
            neg = random.randint(0, matrix.shape[1] - 1)
            if neg not in user_items[u]:
                lst.append(neg)
                count += 1
        all_users.append(u)
        all_lists.append(lst)
    return np.array(all_users), np.array(all_lists)


def compute_enhanced_bpr_loss(student_scores, reg_loss=0.0):
    """Compute BPR loss for listwise data."""
    pos = student_scores[:, 0].unsqueeze(1)
    neg = student_scores[:, 1:]
    return -torch.mean(F.logsigmoid(pos - neg)) + reg_loss


def compute_listwise_distill_loss(s_scores, t_scores, temp=2.0):
    """Compute KL divergence loss for distillation."""
    s_probs = F.log_softmax(s_scores / temp, dim=1)
    t_probs = F.softmax(t_scores / temp, dim=1).detach()
    return F.kl_div(s_probs, t_probs, reduction='batchmean')


def evaluate_lightgcn(model, valid_in, valid_out, device, batch_size=100, ndcg20_only=False):
    """
    Custom evaluation loop for LightGCN using GPU acceleration from utils.py.
    """
    model.eval()
    n_users = valid_in.shape[0]

    if ndcg20_only:
        all_ndcg_values = []
        with torch.no_grad():
            num_batches = (n_users + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_users)

                batch_users = torch.arange(start_idx, end_idx, device=device)
                batch_predictions = model.getUsersRating(batch_users)

                if valid_in is not valid_out:
                    batch_data_in = valid_in[start_idx:end_idx]
                    batch_data_in_dense = torch.FloatTensor(batch_data_in.toarray()).to(device)
                    batch_predictions[batch_data_in_dense > 0] = -float('inf')

                batch_gt = valid_out[start_idx:end_idx]
                batch_gt_dense = torch.FloatTensor(batch_gt.toarray()).to(device)

                batch_results = compute_metrics_gpu(batch_predictions, batch_gt_dense, k_list=[20])

                if 'ndcg@20' in batch_results:
                    all_ndcg_values.extend(batch_results['ndcg@20'].cpu().numpy())

                del batch_predictions, batch_gt_dense, batch_data_in_dense

        return np.mean(all_ndcg_values) if len(all_ndcg_values) > 0 else 0.0

    else:
        k_list = [10, 20, 50]
        metric_names = []
        for k in k_list:
            metric_names.extend([f'ndcg@{k}', f'recall@{k}', f'precision@{k}'])

        all_metric_values = {name: [] for name in metric_names}

        with torch.no_grad():
            num_batches = (n_users + batch_size - 1) // batch_size
            for idx in range(num_batches):
                start, end = idx * batch_size, min((idx + 1) * batch_size, n_users)
                batch_users = torch.arange(start, end, device=device)
                batch_pred = model.getUsersRating(batch_users)

                if valid_in is not valid_out:
                    batch_data_in = valid_in[start:end]
                    batch_data_in_dense = torch.FloatTensor(batch_data_in.toarray()).to(device)
                    batch_pred[batch_data_in_dense > 0] = -float('inf')

                batch_gt = valid_out[start:end]
                batch_gt_dense = torch.FloatTensor(batch_gt.toarray()).to(device)

                batch_results = compute_metrics_gpu(batch_pred, batch_gt_dense, k_list)

                for name, values in batch_results.items():
                    all_metric_values[name].extend(values.cpu().numpy())

                del batch_pred, batch_gt_dense, batch_data_in_dense

        results = {}
        for name, values in all_metric_values.items():
            results[name] = np.mean(values) if len(values) > 0 else 0.0
        return results


def train_listwise_distill_lightgcn(model, optimizer, user_item_matrix, teacher_loader,
                                    valid_in_data, valid_out_data, device, n_epochs,
                                    listwise_K=4, distill_temperature=2.0,
                                    listwise_weight=0.3, batch_size=2048,
                                    early_stopping=50, eval_batch_size=100):
    model.train()
    logging.info("Creating adjacency matrix...")
    adjacency_matrix = model.create_adjacency_matrix(user_item_matrix)
    model.set_adjacency_matrix(adjacency_matrix)
    logging.info("Adjacency matrix created")

    best_ndcg = -np.inf
    best_epoch = 0
    stopping_step = 0
    best_state = None
    best_val_metrics = {}

    for epoch in range(n_epochs):
        model.train()
        user_ids, item_lists = prepare_listwise_training_data(user_item_matrix, K=listwise_K)
        indices = np.random.permutation(len(user_ids))
        user_ids = user_ids[indices]
        item_lists = item_lists[indices]

        n_batches = (len(user_ids) + batch_size - 1) // batch_size

        for idx in range(n_batches):
            start, end = idx * batch_size, min((idx + 1) * batch_size, len(user_ids))
            batch_u = torch.LongTensor(user_ids[start:end]).to(device)
            batch_i = torch.LongTensor(item_lists[start:end]).to(device)

            user_all_emb, item_all_emb = model.computer()

            s_scores = model.compute_listwise_scores(batch_u, batch_i, user_all_emb, item_all_emb)

            t_scores = torch.FloatTensor(
                teacher_loader.get_listwise_teacher_scores(
                    user_ids[start:end], item_lists[start:end]
                )
            ).to(device)

            reg_loss = model.reg_weight * (
                    torch.norm(user_all_emb[batch_u]) ** 2 + torch.norm(item_all_emb[batch_i]) ** 2) / batch_u.size(0)

            bpr = compute_enhanced_bpr_loss(s_scores, reg_loss)
            distill = compute_listwise_distill_loss(s_scores, t_scores, distill_temperature)
            loss = bpr + listwise_weight * distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Evaluating validation at epoch {epoch + 1}...")
            current_ndcg = evaluate_lightgcn(
                model, valid_in_data, valid_out_data, device, eval_batch_size, ndcg20_only=True
            )

            logging.info(f"Epoch {epoch + 1}: Val NDCG@20 = {current_ndcg:.10f}")

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                stopping_step = 0
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = {'ndcg@20': best_ndcg}
            else:
                stopping_step += 5

            if stopping_step >= early_stopping: break

    if best_state:
        model.load_state_dict(best_state)
        logging.info(f"Training finished. Computing full metrics for best model (Epoch {best_epoch})...")
        best_val_metrics = evaluate_lightgcn(
            model, valid_in_data, valid_out_data, device, eval_batch_size, ndcg20_only=False
        )

    return model, best_val_metrics


def precompute_lightgcn_scores(model, train_matrix, device, batch_size, save_dir):
    """
    Precompute LightGCN scores.
    """
    n_users, n_items = train_matrix.shape

    logging.info(f"Precomputing LightGCN scores (Raw output, no replacement)...")

    lightgcn_scores_path = os.path.join(save_dir, 'lightgcn_scores.dat')
    lightgcn_scores = np.memmap(lightgcn_scores_path, dtype='float32', mode='w+', shape=(n_users, n_items))

    model.eval()
    with torch.no_grad():
        num_batches = (n_users + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_users)

            users = torch.arange(start, end, device=device)
            batch_gcn_scores = model.getUsersRating(users).cpu().numpy()

            lightgcn_scores[start:end] = batch_gcn_scores
            lightgcn_scores.flush()

    logging.info(f"LightGCN scores saved to: {lightgcn_scores_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--ratio', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='../data/processed_data')
    parser.add_argument('--bpr_batch', type=int, default=2048)
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--testbatch', type=int, default=2048)
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=800)
    parser.add_argument('--lora_ranks', nargs='+', type=int, default=[4])
    parser.add_argument('--gate_dims', nargs='+', type=int, default=[16])
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--listwise_K', type=int, default=4)
    parser.add_argument('--distill_temperature', type=float, default=4.0)
    parser.add_argument('--listwise_weight', type=float, default=0.3)
    parser.add_argument('--precompute_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    cache_dir = f"precompute_cache/{args.dataset}/{str(args.ratio)}"
    os.makedirs(cache_dir, exist_ok=True)

    timestamp = int(datetime.now().timestamp())
    result_dir = f"distill_gnn_results/{args.dataset}_{args.ratio}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    init_logging(os.path.join(result_dir, 'log.txt'))
    logging.info(f"Configuration: {vars(args)}")

    data = get_open_private_data(os.path.join(args.data_path, args.dataset, str(args.ratio)))
    open_train, open_val_in, open_val_out, _, _ = data[0:5]
    n_users, n_items, user_map = data[10], data[12], data[13]

    lora = f"gate_lora_model/{args.dataset}/{args.ratio}/gated_lora_model.pth"

    logging.info("Step 1: Precomputing Base Scores")
    base_model = load_base_model_for_precompute(lora, [args.latent_dim, args.hidden_dim, n_items],
                                                args.lora_ranks, args.gate_dims, args.lora_dropout,
                                                torch.device(f"cuda:{args.gpu}"))
    precompute_base_scores(base_model, open_train, torch.device(f"cuda:{args.gpu}"), args.precompute_batch_size,
                           cache_dir)
    del base_model
    torch.cuda.empty_cache()

    logging.info("Step 2-3: Precomputing Probs and Features")
    precompute_base_probs_from_scores(cache_dir, n_users, n_items)
    precompute_static_features(open_train, cache_dir)

    logging.info("Step 4: Training LightGCN")
    teacher = TeacherScoreLoader(cache_dir, n_users, n_items)
    model = DistillLightGCN(n_users, n_items, args.recdim, args.layer, args.decay).to(torch.device(f"cuda:{args.gpu}"))
    optim = Adam(model.parameters(), lr=args.lr)

    model, best_val_metrics = train_listwise_distill_lightgcn(
        model=model,
        optimizer=optim,
        user_item_matrix=open_train,
        teacher_loader=teacher,
        valid_in_data=open_val_in,
        valid_out_data=open_val_out,
        device=torch.device(f"cuda:{args.gpu}"),
        n_epochs=args.epochs,
        listwise_K=args.listwise_K,
        distill_temperature=args.distill_temperature,
        listwise_weight=args.listwise_weight,
        batch_size=args.bpr_batch,
        early_stopping=args.early_stopping,
        eval_batch_size=args.testbatch
    )

    logging.info("=== Best Validation Metrics (Full Evaluation) ===")
    for k, v in best_val_metrics.items():
        logging.info(f"  {k}: {v:.10f}")

    logging.info("Step 5-6: Precomputing LightGCN Scores and Uncertainty")
    precompute_lightgcn_scores(model, open_train, torch.device(f"cuda:{args.gpu}"), args.precompute_batch_size,
                               cache_dir)
    precompute_uncertainty_table(cache_dir, n_users, n_items)