import os
import torch
import numpy as np
import argparse
import logging
import copy
import gc
import random
from datetime import datetime
from torch.optim import Adam

from model import MultiVAE
from utils import set_seed, init_logging, evaluate_full_metrics
from data import (get_open_private_data, prepare_user_data_dict, prepare_sparse_batch)


def pretrain_multivae_model(args):
    """
    Pre-train MultiVAE model using only open user data.

    Args:
        args: Parsed command line arguments.
    """

    set_seed(args.seed)

    pretrain_model_dir = "pretrain_model"
    dataset_dir = os.path.join(pretrain_model_dir, args.dataset)
    ratio_dir = os.path.join(dataset_dir, str(args.ratio))
    if not os.path.exists(pretrain_model_dir): os.makedirs(pretrain_model_dir)
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    if not os.path.exists(ratio_dir): os.makedirs(ratio_dir)

    timestamp = int(datetime.now().timestamp())
    result_dir = f"pretrain_results/{args.dataset}_{args.ratio}_{timestamp}"
    if not os.path.exists("pretrain_results"): os.makedirs("pretrain_results")
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    log_filename = os.path.join(result_dir, 'pretrain.log')
    init_logging(log_filename)
    logging.info(f"Configuration: {vars(args)}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset_path = os.path.join(args.data_path, args.dataset, str(args.ratio))
    data = get_open_private_data(dataset_path)
    (open_train_matrix, open_valid_in_data, open_valid_out_data, open_test_in_data, open_test_out_data,
     private_train_matrix, private_valid_in_data, private_valid_out_data, private_test_in_data, private_test_out_data,
     n_open_users, n_private_users, n_items, open_user_map, private_user_map, item_map) = data

    open_user_data = prepare_user_data_dict(open_train_matrix, open_user_map, "open users")

    p_dims = [args.latent_dim, args.hidden_dim, n_items]
    model = MultiVAE(p_dims=p_dims, dropout_p=args.dropout_p).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, best_val_metrics = pretrain(
        model=model,
        optimizer=optimizer,
        open_user_data=open_user_data,
        valid_in_data=open_valid_in_data,
        valid_out_data=open_valid_out_data,
        device=device,
        n_epochs=args.n_epochs,
        anneal_cap=args.anneal_cap,
        early_stopping=args.early_stopping,
        eval_batch_size=args.eval_batch_size,
        batch_size=args.batch_size,
        n_items=n_items
    )

    pretrain_save_path = os.path.join(ratio_dir, 'pretrained_multivae.pth')
    torch.save(model.state_dict(), pretrain_save_path)
    logging.info(f"Final best model saved to: {pretrain_save_path}")

    logging.info("=== Best Validation Metrics ===")
    for k, v in best_val_metrics.items():
        logging.info(f"  {k}: {v:.10f}")

    logging.info("=== Final Evaluation ===")

    logging.info("Evaluating on Open Users Test Set...")
    open_test_results = evaluate_full_metrics(model, open_test_in_data, open_test_out_data, device,
                                              args.eval_batch_size)
    for k, v in open_test_results.items():
        logging.info(f"  {k}: {v:.10f}")

    logging.info("Evaluating on Private Users Test Set...")
    private_test_results = evaluate_full_metrics(model, private_test_in_data, private_test_out_data, device,
                                                 args.eval_batch_size)
    for k, v in private_test_results.items():
        logging.info(f"  {k}: {v:.10f}")

    final_test_results = {'open_test': open_test_results, 'private_test': private_test_results}
    np.save(os.path.join(result_dir, 'final_test_results.npy'), final_test_results)


def pretrain(model, optimizer, open_user_data, valid_in_data, valid_out_data,
                                    device, n_epochs, anneal_cap, early_stopping=50,
                                    eval_batch_size=500, batch_size=500, n_items=None):
    """
    Pretrain. Evaluates full metrics but monitors NDCG@20.
    """
    best_ndcg = -np.inf
    best_epoch = 0
    stopping_step = 0
    best_state = None
    best_val_metrics = {}

    for epoch in range(n_epochs):
        model.train()
        all_users = list(open_user_data.keys())
        random.shuffle(all_users)

        total_batches = (len(all_users) + batch_size - 1) // batch_size
        total_loss = 0.0
        successful_batches = 0

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]

            try:
                batch_input = prepare_sparse_batch(batch_users, open_user_data, n_items, device)
                if batch_input.shape[0] == 0: continue

                optimizer.zero_grad()
                recon_batch, mu, logvar = model(batch_input)
                loss = model.loss_function(recon_batch, batch_input, mu, logvar, anneal_cap)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                successful_batches += 1
                del batch_input, recon_batch, mu, logvar, loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except RuntimeError as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / successful_batches if successful_batches > 0 else float('inf')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Evaluating validation set at epoch {epoch + 1}...")

            # Evaluate all metrics
            val_results = evaluate_full_metrics(model, valid_in_data, valid_out_data, device, eval_batch_size)
            current_ndcg = val_results['ndcg@20']

            # Log only NDCG@20 during training
            logging.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}, Val NDCG@20 = {current_ndcg:.10f}")

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                stopping_step = 0
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = copy.deepcopy(val_results)  # Save full metrics for best epoch
            else:
                stopping_step += 5

            if stopping_step >= early_stopping:
                logging.info("Early stopping triggered.")
                break

            torch.cuda.empty_cache()
            gc.collect()

    if best_state is not None:
        model.load_state_dict(best_state)

    logging.info(f'Pre-training completed. Best NDCG@20: {best_ndcg:.10f} at epoch {best_epoch}')
    return model, best_val_metrics


def main():
    parser = argparse.ArgumentParser(description="Pre-train MultiVAE")
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--data_path', type=str, default='../data/processed_data')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--latent_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--anneal_cap', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--eval_batch_size', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--ratio', type=int, required=True)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    pretrain_multivae_model(args)


if __name__ == "__main__":
    main()