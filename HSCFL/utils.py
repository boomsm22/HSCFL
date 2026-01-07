import logging
import numpy as np
import random
import torch
import bottleneck as bn
import gc


def set_seed(seed):
    """
    Set random seed for reproducible experiments.

    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_logging(log_filename):
    """
    Initialize logging configuration.

    Args:
        log_filename: Path to the log file.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=log_filename,
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def compute_metrics_gpu(pred_scores, ground_truth, k_list=[10, 20, 50]):
    """
    Compute NDCG, Recall, and Precision on GPU.
    """
    results = {}

    max_k = max(k_list)
    _, topk_indices = torch.topk(pred_scores, k=max_k, dim=1)

    topk_labels = torch.gather(ground_truth, 1, topk_indices)

    num_pos = ground_truth.sum(dim=1)

    for k in k_list:
        k_labels = topk_labels[:, :k]

        hits = k_labels.sum(dim=1)

        mask = (num_pos > 0)

        # --- Precision ---
        # Logic: hits / k
        precision_k = hits / k
        results[f'precision@{k}'] = precision_k[mask]

        # --- Recall ---
        # Logic: hits / min(k, num_pos)
        denom_recall = torch.clamp(num_pos, max=k)
        recall_k = hits / (denom_recall + 1e-10)  # epsilon for stability
        results[f'recall@{k}'] = recall_k[mask]

        # --- NDCG ---
        log_positions = torch.log2(torch.arange(2, k + 2, device=pred_scores.device).float())
        dcg = (k_labels / log_positions).sum(dim=1)

        ideal_gains = 1.0 / log_positions
        cumsum_ideal_gains = torch.cumsum(ideal_gains, dim=0)

        idcg_indices = torch.clamp(num_pos, max=k).long() - 1

        idcg_indices = torch.clamp(idcg_indices, min=0)

        idcg = cumsum_ideal_gains[idcg_indices]

        idcg[num_pos == 0] = 0.0

        ndcg_k = dcg / (idcg + 1e-10)
        results[f'ndcg@{k}'] = ndcg_k[mask]

    return results


def evaluate_full_metrics(model, data_in, data_out, device, batch_size=500):
    """
    Evaluate model performance using GPU acceleration.
    Returns dictionary with metrics.
    """
    model.eval()

    k_list = [10, 20, 50]
    metric_names = []
    for k in k_list:
        metric_names.extend([f'ndcg@{k}', f'recall@{k}', f'precision@{k}'])

    all_metric_values = {name: [] for name in metric_names}
    n_users = data_in.shape[0]

    with torch.no_grad():
        num_batches = (n_users + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_users)

            batch_data_in = data_in[start_idx:end_idx]
            batch_data_out = data_out[start_idx:end_idx]

            batch_data_dense = batch_data_in.toarray()
            batch_input_tensor = torch.FloatTensor(batch_data_dense).to(device)

            if hasattr(model, 'forward'):
                output = model(batch_input_tensor)
                if isinstance(output, tuple):
                    batch_predictions = output[0]
                else:
                    batch_predictions = output
            else:
                batch_predictions = model(batch_input_tensor)

            batch_predictions[batch_input_tensor > 0] = -float('inf')

            batch_gt_dense = batch_data_out.toarray()
            batch_gt_tensor = torch.FloatTensor(batch_gt_dense).to(device)

            batch_results = compute_metrics_gpu(batch_predictions, batch_gt_tensor, k_list)

            for name, values in batch_results.items():
                all_metric_values[name].extend(values.cpu().numpy())

            del batch_input_tensor, batch_predictions, batch_gt_tensor

    results = {}
    for name in metric_names:
        values = all_metric_values[name]
        results[name] = np.mean(values) if len(values) > 0 else 0.0

    return results