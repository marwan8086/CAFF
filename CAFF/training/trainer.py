import os
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.bce import bce_loss
from losses.depth_contrastive import depth_contrastive_loss
from losses.hc3 import hc3_loss
from training.metrics import compute_masked_metrics
from data.dataset import collate_fn


class Trainer:
    def __init__(self, model, train_dataset, dev_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.save_dir = config['save_dir']
        self.theta = config['theta']

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(config['epochs'] - config['warmup_epochs'], 1),
        )
        self.best_metric = 0.0
        os.makedirs(self.save_dir, exist_ok=True)

    def _build_retained_relation_ids(self, candidate_rel_ids: List[torch.Tensor], labels: List[torch.Tensor]):
        retained = []
        for ell in range(self.model.L):
            batch_retained = []
            rel_ids = candidate_rel_ids[ell]
            label_hop = labels[ell]
            for i in range(rel_ids.shape[0]):
                valid_mask = rel_ids[i] >= 0
                retained_ids = rel_ids[i][valid_mask & (label_hop[i] == 1.0)]
                batch_retained.append(retained_ids)
            retained.append(batch_retained)
        return retained

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Train Epoch {epoch}')):
            queries = batch['queries']
            if not queries:  # Skip empty batches after quality filtering
                continue
            candidate_rel_ids = [x.to(self.device) for x in batch['candidate_rel_ids']]
            labels = [x.to(self.device) for x in batch['labels']]
            masks = [x.to(self.device) for x in batch['masks']]

            retained_relation_ids = self._build_retained_relation_ids(candidate_rel_ids, labels)
            self.optimizer.zero_grad()

            logits_by_hop = self.model(queries, candidate_rel_ids, retained_relation_ids)
            loss_bce = 0.0
            for ell in range(self.model.L):
                loss_bce += bce_loss(logits_by_hop[ell], labels[ell], masks[ell].float())
            loss_bce = loss_bce / self.model.L

            loss_dc = depth_contrastive_loss(candidate_rel_ids, logits_by_hop, labels, masks, self.config['gamma_D'])
            loss_hc3 = hc3_loss(
                self.model,
                queries,
                candidate_rel_ids,
                labels,
                masks,
                retained_relation_ids,
                gamma_C=self.config['gamma_C'],
            )

            loss = loss_bce + self.config['lambda_D'] * loss_dc + self.config['lambda_C'] * loss_hc3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            epoch_loss += loss.item()

        if epoch >= self.config['warmup_epochs']:
            self.scheduler.step()
        return epoch_loss / len(self.train_loader)

    def infer_with_feedback(self, batch):
        queries = batch['queries']
        candidate_rel_ids = [x.to(self.device) for x in batch['candidate_rel_ids']]
        masks = [x.to(self.device) for x in batch['masks']]

        batch_size = candidate_rel_ids[0].shape[0]
        q_embed = self.model.encode_queries(queries)
        z_prev = torch.zeros(batch_size, self.model.d_model, device=self.device)
        scores_by_hop = []
        retained_relation_ids = []

        for ell in range(self.model.L):
            logits = self.model.score_candidates(q_embed, candidate_rel_ids[ell], z_prev, ell)
            scores = torch.sigmoid(logits)
            scores_by_hop.append(scores.detach().cpu().numpy())

            retained_ids = []
            for i in range(batch_size):
                keep_mask = (scores[i] >= self.theta) & masks[ell][i]
                rels = candidate_rel_ids[ell][i][keep_mask]
                retained_ids.append(rels)
            retained_relation_ids.append(retained_ids)
            if ell < self.model.L - 1:
                z_prev = self.model.csv_encoder(retained_ids, self.model.rel_embeddings)

        return scores_by_hop, retained_relation_ids

    def evaluate(self):
        self.model.eval()
        all_scores = [[] for _ in range(self.model.L)]
        all_labels = [[] for _ in range(self.model.L)]
        all_masks = [[] for _ in range(self.model.L)]

        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc='Validation'):
                if not batch['queries']:  # Skip empty batches
                    continue
                scores_by_hop, _ = self.infer_with_feedback(batch)
                for ell in range(self.model.L):
                    all_scores[ell].append(scores_by_hop[ell])
                    all_labels[ell].append(batch['labels'][ell].numpy())
                    all_masks[ell].append(batch['masks'][ell].numpy())

        flattened_scores = [np.concatenate(x, axis=0) for x in all_scores]
        flattened_labels = [np.concatenate(x, axis=0) for x in all_labels]
        flattened_masks = [np.concatenate(x, axis=0) for x in all_masks]
        return compute_masked_metrics(flattened_scores, flattened_labels, flattened_masks, threshold=self.theta)

    def run(self):
        import numpy as np
        for epoch in range(1, self.config['epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            print(f"Epoch {epoch}: loss={train_loss:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, map={metrics['map']:.4f}")

            if metrics['f1'] > self.best_metric:
                self.best_metric = metrics['f1']
                save_path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
