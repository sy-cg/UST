import os

import ust_bootstrap  # noqa: F401
import torch
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.utils import InputType

from ust_core_v2 import USTV2Tokenizer, USTv2Fusion
from ust_utils import (
    build_frozen_embedding,
    build_item_domain_tensor,
    get_processed_dir,
    pairwise_domain_masks,
)


class USTv2_SASRec(SASRec):
    """Improved USTv2 variant for sequential recommendation."""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.beta1 = config["beta1"] if "beta1" in config else 0.01
        self.beta2 = config["beta2"] if "beta2" in config else 0.05
        self.beta3 = config["beta3"] if "beta3" in config else 0.05
        self.beta4 = config["beta4"] if "beta4" in config else 0.1
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

        self.eval_test_zi = None
        self.train_all_zi_no_grad = None

    def _load_multimodal_features(self, config):
        dataset_name = config["dataset"]
        text_path = os.path.join(self.processed_dir, "text_features.npy")
        vision_path = os.path.join(self.processed_dir, "vision_features.npy")

        self.text_embedding = build_frozen_embedding(text_path, config["device"])
        self.vision_embedding = build_frozen_embedding(vision_path, config["device"])
        print(f"[USTv2] Loaded multimodal features for {dataset_name} from {self.processed_dir}", flush=True)

    def get_fused_embedding(self, item_ids):
        id_emb = self.item_embedding(item_ids)
        text_feat = self.text_embedding(item_ids)
        vision_feat = self.vision_embedding(item_ids)
        fused_emb, _ = self.fusion(id_emb, text_feat, vision_feat)
        return fused_emb

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

    def forward(self, item_seq, item_seq_len, advance_token_step=False):
        fused_emb = self.get_fused_embedding(item_seq)
        z_i, _, _, loss_balance, loss_orth, loss_commit = self.ust_tokenizer(
            fused_emb,
            use_ust=self.use_ust,
            advance_step=advance_token_step,
            ust_mode=self.ust_token_mode,
        )

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = z_i + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output, loss_balance, loss_orth, loss_commit

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output, _, _, _ = self.forward(item_seq, item_seq_len, advance_token_step=True)

        if self.train_all_zi_no_grad is None:
            with torch.no_grad():
                all_items_idx = torch.arange(self.n_items, device=item_seq.device)
                all_item_emb = self.get_fused_embedding(all_items_idx)
                all_zi_no_grad, _, _, _, _, _ = self.ust_tokenizer(
                    all_item_emb,
                    use_ust=self.use_ust,
                    advance_step=False,
                    ust_mode=self.ust_token_mode,
                )
                self.train_all_zi_no_grad = all_zi_no_grad.detach()

        all_zi_ste = (
            self.train_all_zi_no_grad
            + self.item_embedding.weight
            - self.item_embedding.weight.detach()
        )

        batch_items = torch.cat([item_seq.reshape(-1), pos_items.reshape(-1)])
        unique_items = torch.unique(batch_items)
        unique_items = unique_items[unique_items > 0]
        batch_emb = self.get_fused_embedding(unique_items)
        batch_zi_grad, batch_es_grad, _, batch_loss_balance, batch_loss_orth, batch_loss_commit = self.ust_tokenizer(
            batch_emb,
            use_ust=self.use_ust,
            advance_step=False,
            ust_mode=self.ust_token_mode,
        )

        mask = torch.zeros(self.n_items, 1, device=item_seq.device, dtype=torch.bool)
        mask[unique_items] = True

        full_grad_zi = torch.zeros_like(all_zi_ste)
        full_grad_zi[unique_items] = batch_zi_grad
        test_zi = torch.where(mask, full_grad_zi, all_zi_ste)

        logits = torch.matmul(seq_output, test_zi.transpose(0, 1))
        loss_rec = self.loss_fct(logits, pos_items)
        loss_align = self._compute_alignment_loss(batch_es_grad, unique_items)

        return (
            loss_rec
            + self.beta1 * batch_loss_balance
            + self.beta2 * loss_align
            + self.beta3 * batch_loss_orth
            + self.beta4 * batch_loss_commit
        )

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output, _, _, _ = self.forward(item_seq, item_seq_len, advance_token_step=False)
        test_item_emb = self.get_fused_embedding(test_item.unsqueeze(1)).squeeze(1)
        test_zi, _, _, _, _, _ = self.ust_tokenizer(
            test_item_emb,
            use_ust=self.use_ust,
            advance_step=False,
            ust_mode=self.ust_token_mode,
        )
        return torch.mul(seq_output, test_zi).sum(dim=1)

    def train(self, mode=True):
        if mode:
            self.eval_test_zi = None
            self.train_all_zi_no_grad = None
        return super().train(mode)

    def load_other_parameter(self, para):
        super().load_other_parameter(para)
        self.eval_test_zi = None
        self.train_all_zi_no_grad = None

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _, _ = self.forward(item_seq, item_seq_len, advance_token_step=False)

        if self.eval_test_zi is None:
            test_items_emb = self.get_fused_embedding(
                torch.arange(self.n_items, device=item_seq.device)
            )
            self.eval_test_zi, _, _, _, _, _ = self.ust_tokenizer(
                test_items_emb,
                use_ust=self.use_ust,
                advance_step=False,
                ust_mode=self.ust_token_mode,
            )

        return torch.matmul(seq_output, self.eval_test_zi.transpose(0, 1))
