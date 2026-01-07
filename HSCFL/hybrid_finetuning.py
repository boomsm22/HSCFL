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

from model import MultiVAE
from utils import set_seed, init_logging, evaluate_full_metrics
from data import (get_open_private_data, prepare_all_user_data_dict, prepare_sparse_batch,
                  merge_validation_data, merge_test_data)


class GatedLoRALayer(nn.Module):
    """
    Gated LoRA Layer.
    Combines original weights and LoRA adapter weights using a learned gate.
    """

    def __init__(self, in_features, out_features, rank=4, gate_dim=16, dropout_p=0.1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_scaling = 1.0 / rank
        self.gate_network = nn.Sequential(
            nn.Linear(in_features, gate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(gate_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout_p)
        nn.init.kaiming_normal_(self.lora_A, mode='fan_in', nonlinearity='tanh')

    def forward(self, x, original_output):
        lora_output = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.lora_scaling
        gate_weights = self.gate_network(x)
        w_original = gate_weights[:, 0:1]
        w_lora = gate_weights[:, 1:2]
        return w_original * original_output + w_lora * lora_output


class GatedLoRAMultiVAE(nn.Module):
    """
    Gated LoRA MultiVAE Model.
    Fine-tunes a pre-trained MultiVAE using Gated LoRA on both open and private data.
    """

    def __init__(self, pretrained_model, lora_ranks, gate_dims=None, lora_dropout=0.1):
        super().__init__()
        self.p_dims = pretrained_model.p_dims
        self.q_dims = pretrained_model.q_dims
        with torch.no_grad():
            for i, layer in enumerate(pretrained_model.q_layers):
                self.register_buffer(f'q_weight_{i}', layer.weight.data.clone())
                self.register_buffer(f'q_bias_{i}', layer.bias.data.clone())
            for i, layer in enumerate(pretrained_model.p_layers):
                self.register_buffer(f'p_weight_{i}', layer.weight.data.clone())
                self.register_buffer(f'p_bias_{i}', layer.bias.data.clone())
        del pretrained_model
        torch.cuda.empty_cache()
        gc.collect()
        self.drop = nn.Dropout(0.5)
        num_enc_layers = len(self.q_dims) - 1
        if isinstance(lora_ranks, int): lora_ranks = [lora_ranks] * num_enc_layers
        if gate_dims is None:
            gate_dims = [16] * len(lora_ranks)
        elif isinstance(gate_dims, int):
            gate_dims = [gate_dims] * len(lora_ranks)
        self.encoder_gated_lora = nn.ModuleList()
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        for i, (rank, g_dim) in enumerate(zip(lora_ranks, gate_dims)):
            if rank > 0:
                self.encoder_gated_lora.append(
                    GatedLoRALayer(temp_q_dims[i], temp_q_dims[i + 1], rank, g_dim, lora_dropout))
            else:
                self.encoder_gated_lora.append(None)
        self.decoder_gated_lora = nn.ModuleList()
        dec_ranks = lora_ranks[::-1]
        dec_gate_dims = gate_dims[::-1]
        for i, (rank, g_dim) in enumerate(zip(dec_ranks, dec_gate_dims)):
            if rank > 0:
                self.decoder_gated_lora.append(
                    GatedLoRALayer(self.p_dims[i], self.p_dims[i + 1], rank, g_dim, lora_dropout))
            else:
                self.decoder_gated_lora.append(None)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encoder(self, x):
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        for i, gated_lora in enumerate(self.encoder_gated_lora):
            w = getattr(self, f'q_weight_{i}')
            b = getattr(self, f'q_bias_{i}')
            h_orig = F.linear(h, w, b)
            h = gated_lora(h, h_orig) if gated_lora else h_orig
            if i != len(self.encoder_gated_lora) - 1:
                h = torch.tanh(h)
            else:
                return h[:, :self.q_dims[-1]], h[:, self.q_dims[-1]:]

    def decoder(self, z):
        h = z
        for i, gated_lora in enumerate(self.decoder_gated_lora):
            w = getattr(self, f'p_weight_{i}')
            b = getattr(self, f'p_bias_{i}')
            h_orig = F.linear(h, w, b)
            h = gated_lora(h, h_orig) if gated_lora else h_orig
            if i != len(self.decoder_gated_lora) - 1: h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        return mu

    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        neg_ll = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return neg_ll + anneal * KLD

    def get_lora_and_gate_parameters(self):
        params = []
        for m in self.encoder_gated_lora:
            if m: params.extend(list(m.parameters()))
        for m in self.decoder_gated_lora:
            if m: params.extend(list(m.parameters()))
        return params


def hybrid_finetuning(model, optimizer, all_user_data, valid_in_data, valid_out_data,
                                                       device, n_epochs, anneal_cap, batch_size=500,
                                                       early_stopping=50, eval_batch_size=500, n_items=40981):
    best_ndcg = -np.inf
    best_epoch = 0
    stopping_step = 0
    best_state = None
    best_val_metrics = {}

    logging.info(f"Total users for training: {len(all_user_data)}")
    logging.info(f"Training batch size: {batch_size}")

    for epoch in range(n_epochs):
        logging.info(f"=== Epoch {epoch + 1}/{n_epochs} ===")
        model.train()
        all_users = list(all_user_data.keys())
        random.shuffle(all_users)

        total_batches = (len(all_users) + batch_size - 1) // batch_size
        total_loss = 0.0
        successful_batches = 0

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]
            batch_data = {user_id: all_user_data[user_id] for user_id in batch_users}

            try:
                batch_input = prepare_sparse_batch(batch_users, batch_data, n_items, device)
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
        logging.info(f"Avg Training Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info("Evaluating on merged validation set...")
            val_results = evaluate_full_metrics(model, valid_in_data, valid_out_data, device, eval_batch_size)
            current_ndcg = val_results['ndcg@20']

            logging.info(f"Epoch {epoch + 1}: Val NDCG@20 = {current_ndcg:.10f}")

            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                stopping_step = 0
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = copy.deepcopy(val_results)
            else:
                stopping_step += 5

            if stopping_step >= early_stopping:
                logging.info("Early stopping triggered.")
                break
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    logging.info(f'Hybrid fine-tuning completed. Best NDCG@20: {best_ndcg:.10f} at epoch {best_epoch}')
    return model, best_val_metrics


def main():
    parser = argparse.ArgumentParser(description="Hybrid Fine-tuning with Gated LoRA")
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--data_path', type=str, default='../data/processed_data')
    parser.add_argument('--pretrained_model_path', type=str, required=False)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--latent_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=800)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--anneal_cap', type=float, default=1.0)
    parser.add_argument('--lora_ranks', nargs='+', type=int, default=[4])
    parser.add_argument('--gate_dims', nargs='+', type=int, default=[16])
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--eval_batch_size', type=int, default=500)
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ratio', type=int, required=True)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    gate_lora_model_dir = "gate_lora_model"
    dataset_dir = os.path.join(gate_lora_model_dir, args.dataset)
    ratio_dir = os.path.join(dataset_dir, str(args.ratio))
    if not os.path.exists(gate_lora_model_dir): os.makedirs(gate_lora_model_dir)
    if not os.path.exists(dataset_dir): os.makedirs(dataset_dir)
    if not os.path.exists(ratio_dir): os.makedirs(ratio_dir)

    timestamp = int(datetime.now().timestamp())
    result_dir = f"gated_lora_results/{args.dataset}_{args.ratio}_{timestamp}"
    if not os.path.exists("gated_lora_results"): os.makedirs("gated_lora_results")
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    init_logging(os.path.join(result_dir, 'gated_lora_training.log'))
    logging.info(f"Configuration: {vars(args)}")

    dataset_path = os.path.join(args.data_path, args.dataset, str(args.ratio))
    data = get_open_private_data(dataset_path)
    open_train, open_val_in, open_val_out, open_test_in, open_test_out = data[0:5]
    priv_train, priv_val_in, priv_val_out, priv_test_in, priv_test_out = data[5:10]
    n_items, open_user_map, priv_user_map = data[12], data[13], data[14]

    all_user_data = prepare_all_user_data_dict(open_train, priv_train, open_user_map, priv_user_map, use_prefix=True)
    merged_val_in, merged_val_out = merge_validation_data(open_val_in, open_val_out, priv_val_in, priv_val_out)

    pretrained_path = args.pretrained_model_path or f"pretrain_model/{args.dataset}/{args.ratio}/pretrained_multivae.pth"
    pretrained_model = MultiVAE([args.latent_dim, args.hidden_dim, n_items], args.dropout_p)
    pretrained_model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    model = GatedLoRAMultiVAE(pretrained_model, args.lora_ranks, args.gate_dims, args.lora_dropout).to(device)
    del pretrained_model
    optimizer = Adam(model.get_lora_and_gate_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, best_val_metrics = hybrid_finetuning(
        model, optimizer, all_user_data, merged_val_in, merged_val_out, device,
        args.n_epochs, args.anneal_cap, args.batch_size, args.early_stopping,
        args.eval_batch_size, n_items
    )

    # === Save Best Model to Persistent Directory ===
    gate_lora_save_path = os.path.join(ratio_dir, 'gated_lora_model.pth')
    torch.save(model.state_dict(), gate_lora_save_path)
    logging.info(f"Final best Gate LoRA model saved to: {gate_lora_save_path}")

    # Print best validation metrics
    logging.info("=== Best Validation Metrics ===")
    for k, v in best_val_metrics.items():
        logging.info(f"  {k}: {v:.10f}")

    logging.info("=== Final Evaluation ===")
    merged_test_in, merged_test_out = merge_test_data(open_test_in, open_test_out, priv_test_in, priv_test_out)

    res_total = evaluate_full_metrics(model, merged_test_in, merged_test_out, device, args.eval_batch_size)
    logging.info("Total Test Results:")
    for k, v in res_total.items(): logging.info(f"  {k}: {v:.10f}")

    res_open = evaluate_full_metrics(model, open_test_in, open_test_out, device, args.eval_batch_size)
    logging.info("Open Users Test Results:")
    for k, v in res_open.items(): logging.info(f"  {k}: {v:.10f}")

    res_priv = evaluate_full_metrics(model, priv_test_in, priv_test_out, device, args.eval_batch_size)
    logging.info("Private Users Test Results:")
    for k, v in res_priv.items(): logging.info(f"  {k}: {v:.10f}")


if __name__ == "__main__":
    main()