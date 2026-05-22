import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import ust_bootstrap  # noqa: F401
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.utils import InputType

from ust_core_v2 import USTV2Tokenizer, USTv2Fusion
from ust_utils import (
    build_frozen_embedding,
    build_item_domain_tensor,
    get_processed_dir,
    pairwise_domain_masks,
)


class USTv2_LightGCN(LightGCN):
    """Improved USTv2 variant for graph recommendation."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = config["embedding_size"]
        self.beta1 = config["beta1"] if "beta1" in config else 0.01
        self.beta2 = config["beta2"] if "beta2" in config else 0.05
        self.beta3 = config["beta3"] if "beta3" in config else 0.05
        self.beta4 = config["beta4"] if "beta4" in config else 0.1
        self.mm_loss_weight = config["mm_loss_weight"] if "mm_loss_weight" in config else 0.1
        self.use_ust = config["use_ust"] if "use_ust" in config else True
        self.ust_token_mode = config["ust_token_mode"] if "ust_token_mode" in config else "full"

        vocab_size = config["vocab_size"] if "vocab_size" in config else 256
        gumbel_tau = config["gumbel_tau"] if "gumbel_tau" in config else 1.0
        tau_min = config["tau_min"] if "tau_min" in config else 0.2
        fusion_dropout = config["fusion_dropout"] if "fusion_dropout" in config else 0.1

        self.processed_dir = get_processed_dir(config)
        self._load_multimodal_features(config)
        self.item_domains = build_item_domain_tensor(
            dataset=dataset,
            item_field=self.ITEM_ID,
            processed_dir=self.processed_dir,
            device=self.device,
        )

        self.fusion = USTv2Fusion(hidden_size=self.hidden_size, dropout=fusion_dropout)
        self.ust_tokenizer = USTV2Tokenizer(
            input_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            vocab_size=vocab_size,
            tau_init=gumbel_tau,
            tau_min=tau_min,
        )

        self.train_item_zi_no_grad = None
        self.eval_item_zi = None
        self.current_graph_item_ids = None
        self.current_graph_advance = False
        self.latest_graph_token_state = None

    def _load_multimodal_features(self, config):
        dataset_name = config["dataset"]
        text_path = os.path.join(self.processed_dir, "text_features.npy")
        vision_path = os.path.join(self.processed_dir, "vision_features.npy")

        self.text_embedding = build_frozen_embedding(text_path, config["device"])
        self.vision_embedding = build_frozen_embedding(vision_path, config["device"])
        print(
            f"[USTv2] Loaded multimodal features for {dataset_name} from {self.processed_dir}",
            flush=True,
        )

    def get_fused_embedding(self, item_ids):
        id_emb = self.item_embedding(item_ids)
        text_feat = self.text_embedding(item_ids)
        vision_feat = self.vision_embedding(item_ids)
        fused_emb, _ = self.fusion(id_emb, text_feat, vision_feat)
        return fused_emb

    def _tokenize_items(self, item_ids, advance_step=False):
        fused_emb = self.get_fused_embedding(item_ids)
        z_i, e_s, e_p, loss_balance, loss_orth, loss_commit = self.ust_tokenizer(
            fused_emb,
            use_ust=self.use_ust,
            advance_step=advance_step,
            ust_mode=self.ust_token_mode,
        )
        return fused_emb, z_i, e_s, e_p, loss_balance, loss_orth, loss_commit

    def _compute_lightgcn_orth_loss(self, shared_embeddings, private_embeddings):
        if not self.use_ust or self.beta3 <= 0:
            return torch.tensor(0.0, device=self.device)

        shared_norm = F.normalize(shared_embeddings, p=2, dim=-1)
        private_norm = F.normalize(private_embeddings, p=2, dim=-1)
        return torch.sum(shared_norm * private_norm, dim=-1).pow(2).mean()

    def _compute_lightgcn_commit_loss(self, discrete_embeddings, fused_embeddings):
        if not self.use_ust or self.beta4 <= 0:
            return torch.tensor(0.0, device=self.device)

        discrete_norm = F.normalize(discrete_embeddings, p=2, dim=-1)
        fused_target = F.normalize(fused_embeddings.detach(), p=2, dim=-1)
        return (1.0 - torch.sum(discrete_norm * fused_target, dim=-1)).mean()

    def _compute_alignment_loss(self, shared_embeddings, item_ids):
        if not self.use_ust or self.beta2 <= 0:
            return torch.tensor(0.0, device=item_ids.device)

        mask_bundle = pairwise_domain_masks(item_ids, self.item_domains)
        if mask_bundle is None:
            return torch.tensor(0.0, device=item_ids.device)

        reference_mask, other_masks = mask_bundle
        reference_embeddings = shared_embeddings[reference_mask]
        losses = []
        for other_mask in other_masks:
            other_embeddings = shared_embeddings[other_mask]
            if reference_embeddings.numel() == 0 or other_embeddings.numel() == 0:
                continue
            losses.append(
                self.ust_tokenizer.compute_distribution_alignment(
                    reference_embeddings, other_embeddings
                )
            )

        if not losses:
            return torch.tensor(0.0, device=item_ids.device)
        return torch.stack(losses).mean()

    def get_norm_adj_mat(self):
        inter_m = self.interaction_matrix
        inter_m_t = self.interaction_matrix.transpose()

        adj = sp.vstack(
            [
                sp.hstack([sp.coo_matrix((self.n_users, self.n_users)), inter_m]),
                sp.hstack([inter_m_t, sp.coo_matrix((self.n_items, self.n_items))]),
            ]
        ).astype(np.float32)

        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum + 1e-7, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()

        edge_index = torch.stack(
            [
                torch.from_numpy(norm_adj.row).long(),
                torch.from_numpy(norm_adj.col).long(),
            ]
        )
        edge_weight = torch.from_numpy(norm_adj.data).float()
        return torch.sparse_coo_tensor(
            edge_index, edge_weight, torch.Size(norm_adj.shape), device=self.device
        ).coalesce()

    def _get_train_item_cache(self):
        if self.train_item_zi_no_grad is None:
            with torch.no_grad():
                all_item_ids = torch.arange(self.n_items, device=self.device)
                _, item_zi_no_grad, _, _, _, _, _ = self._tokenize_items(
                    all_item_ids, advance_step=False
                )
                self.train_item_zi_no_grad = item_zi_no_grad.detach()
        return self.train_item_zi_no_grad

    def _get_eval_item_cache(self):
        if self.eval_item_zi is None:
            with torch.no_grad():
                all_item_ids = torch.arange(self.n_items, device=self.device)
                _, eval_item_zi, _, _, _, _, _ = self._tokenize_items(
                    all_item_ids, advance_step=False
                )
                self.eval_item_zi = eval_item_zi.detach()
        return self.eval_item_zi

    def _build_training_item_embeddings(self):
        cached_item_zi = self._get_train_item_cache()
        item_embeddings = (
            cached_item_zi
            + self.item_embedding.weight
            - self.item_embedding.weight.detach()
        )

        self.latest_graph_token_state = None
        if self.current_graph_item_ids is None or self.current_graph_item_ids.numel() == 0:
            return item_embeddings

        live_item_ids = torch.unique(self.current_graph_item_ids, sorted=False)
        live_fused, live_zi, live_es, live_ep, loss_balance, _, _ = self._tokenize_items(
            live_item_ids, advance_step=self.current_graph_advance
        )
        zero_loss = live_fused.new_tensor(0.0)
        if self.use_ust:
            discrete_embeddings = self.ust_tokenizer._resolve_discrete_representation(
                live_es, live_ep, self.ust_token_mode
            )
            loss_orth = self._compute_lightgcn_orth_loss(live_es, live_ep)
            loss_commit = self._compute_lightgcn_commit_loss(discrete_embeddings, live_fused)
        else:
            loss_orth = zero_loss
            loss_commit = zero_loss

        # Replace the current batch items with live USTv2 representations so the
        # graph-side BPR loss can update the tokenizer directly.
        item_embeddings = item_embeddings.clone()
        item_embeddings[live_item_ids] = live_zi
        self.latest_graph_token_state = {
            "item_ids": live_item_ids,
            "z_i": live_zi,
            "fused": live_fused,
            "shared": live_es,
            "private": live_ep,
            "loss_balance": loss_balance,
            "loss_orth": loss_orth,
            "loss_commit": loss_commit,
        }
        return item_embeddings

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        if self.training:
            item_embeddings = self._build_training_item_embeddings()
        else:
            item_embeddings = self._get_eval_item_cache()
        return torch.cat([user_embeddings, item_embeddings], dim=0)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        batch_item_ids = torch.cat([pos_item.reshape(-1), neg_item.reshape(-1)], dim=0)
        unique_item_ids, inverse_indices = torch.unique(
            batch_item_ids, sorted=False, return_inverse=True
        )
        pos_inverse = inverse_indices[: pos_item.numel()]
        neg_inverse = inverse_indices[pos_item.numel() :]

        self.current_graph_item_ids = unique_item_ids
        self.current_graph_advance = True
        try:
            user_all_embeddings, item_all_embeddings = self.forward()
        finally:
            self.current_graph_item_ids = None
            self.current_graph_advance = False

        token_state = self.latest_graph_token_state

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )
        loss_rec_graph = mf_loss + self.reg_weight * reg_loss

        zero_loss = torch.tensor(0.0, device=user.device)
        loss_rec_mm = zero_loss
        loss_align = zero_loss
        batch_loss_balance = zero_loss
        batch_loss_orth = zero_loss
        batch_loss_commit = zero_loss

        if token_state is not None and self.use_ust:
            batch_loss_balance = token_state["loss_balance"]
            batch_loss_orth = token_state["loss_orth"]
            batch_loss_commit = token_state["loss_commit"]
            loss_align = self._compute_alignment_loss(
                token_state["shared"], token_state["item_ids"]
            )

            if self.mm_loss_weight > 0:
                local_item_z = token_state["z_i"]
                pos_local = local_item_z[pos_inverse]
                neg_local = local_item_z[neg_inverse]
                pos_scores_mm = torch.mul(u_embeddings, pos_local).sum(dim=1)
                neg_scores_mm = torch.mul(u_embeddings, neg_local).sum(dim=1)
                loss_rec_mm = self.mf_loss(pos_scores_mm, neg_scores_mm)

        return (
            loss_rec_graph
            + self.mm_loss_weight * loss_rec_mm
            + self.beta1 * batch_loss_balance
            + self.beta2 * loss_align
            + self.beta3 * batch_loss_orth
            + self.beta4 * batch_loss_commit
        )

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def train(self, mode=True):
        self.restore_user_e = None
        self.restore_item_e = None
        self.latest_graph_token_state = None
        self.current_graph_item_ids = None
        self.current_graph_advance = False
        if mode:
            self.train_item_zi_no_grad = None
        else:
            self.eval_item_zi = None
        return super().train(mode)

    def load_other_parameter(self, para):
        super().load_other_parameter(para)
        self.restore_user_e = None
        self.restore_item_e = None
        self.train_item_zi_no_grad = None
        self.eval_item_zi = None
        self.latest_graph_token_state = None
        self.current_graph_item_ids = None
        self.current_graph_advance = False

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embeddings = self.restore_user_e[user]
        return torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
