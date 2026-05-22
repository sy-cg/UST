import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class USTv2Fusion(nn.Module):
    """A gated multimodal fusion block with per-item modality weighting."""

    def __init__(self, hidden_size, text_dim=768, vision_dim=512, dropout=0.1):
        super().__init__()
        self.id_norm = nn.LayerNorm(hidden_size)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, id_emb, text_feat, vision_feat):
        id_hidden = self.id_norm(id_emb)
        text_hidden = self.text_proj(text_feat)
        vision_hidden = self.vision_proj(vision_feat)

        gate_logits = self.gate(torch.cat([id_hidden, text_hidden, vision_hidden], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)

        fused = (
            gate_weights[..., 0:1] * id_hidden
            + gate_weights[..., 1:2] * text_hidden
            + gate_weights[..., 2:3] * vision_hidden
        )
        return fused, gate_weights


class USTV2Tokenizer(nn.Module):
    """
    Improved USTv2 tokenizer:
    - deterministic inference
    - batch-level code balance instead of per-sample uniformity
    - residual blending between continuous and discrete semantics
    - distribution alignment on shared codes
    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        vocab_size=256,
        tau_init=1.0,
        tau_min=0.2,
        tau_decay=0.9995,
        orth_margin=0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.global_step = 0
        self.orth_margin = orth_margin

        self.shared_logits_proj = nn.Linear(input_dim, vocab_size)
        self.private_logits_proj = nn.Linear(input_dim, vocab_size)

        self.shared_codebook = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.private_codebook = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.shared_codebook)
        nn.init.xavier_uniform_(self.private_codebook)

    def get_current_tau(self):
        if self.training:
            return max(self.tau_min, self.tau_init * (self.tau_decay ** self.global_step))
        return self.tau_min

    def set_tau_min(self, tau_min):
        self.tau_min = float(tau_min)

    def _sample_tokens(self, logits, tau):
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        token_ids = torch.argmax(logits, dim=-1)
        return F.one_hot(token_ids, num_classes=self.vocab_size).float()

    def _code_balance_loss(self, probs):
        flat_probs = probs.reshape(-1, probs.size(-1))
        avg_probs = flat_probs.mean(dim=0)
        balance_loss = torch.sum(
            avg_probs * (torch.log(avg_probs + 1e-12) - math.log(1.0 / self.vocab_size))
        )
        entropy_loss = -torch.sum(flat_probs * torch.log(flat_probs + 1e-12), dim=-1).mean()
        return balance_loss + 0.1 * entropy_loss

    def _orthogonality_loss(self, shared_emb, private_emb):
        shared_norm = F.normalize(shared_emb, p=2, dim=-1).reshape(-1, shared_emb.size(-1))
        private_norm = F.normalize(private_emb, p=2, dim=-1).reshape(-1, private_emb.size(-1))
        overlap = torch.matmul(shared_norm.t(), private_norm) / max(shared_norm.size(0), 1)
        raw_penalty = torch.linalg.norm(overlap, ord="fro") ** 2
        return F.relu(raw_penalty - self.orth_margin)

    def _resolve_discrete_representation(self, e_s, e_p, ust_mode):
        if ust_mode == "shared_only":
            return e_s
        if ust_mode == "private_only":
            return e_p
        return e_s + e_p

    def _compute_outputs(self, x, use_ust=True, advance_step=False, ust_mode="full"):
        if not use_ust:
            zero_loss = torch.tensor(0.0, device=x.device)
            return {
                "z_i": x,
                "e_s": x,
                "e_p": x,
                "loss_balance": zero_loss,
                "loss_orth": zero_loss,
                "loss_commit": zero_loss,
                "tokens_s": None,
                "tokens_p": None,
                "probs_s": None,
                "probs_p": None,
                "discrete_repr": x,
                "continuous_repr": x,
                "tau": self.get_current_tau(),
            }

        if self.training and advance_step:
            self.global_step += 1

        logits_s = self.shared_logits_proj(x)
        logits_p = self.private_logits_proj(x)
        probs_s = F.softmax(logits_s, dim=-1)
        probs_p = F.softmax(logits_p, dim=-1)

        tau = self.get_current_tau()
        tokens_s = self._sample_tokens(logits_s, tau)
        tokens_p = self._sample_tokens(logits_p, tau)

        e_s = torch.matmul(tokens_s, self.shared_codebook)
        e_p = torch.matmul(tokens_p, self.private_codebook)
        discrete_repr = self._resolve_discrete_representation(e_s, e_p, ust_mode)

        mix_gate = torch.sigmoid(self.output_gate(torch.cat([x, discrete_repr], dim=-1)))
        z_i = mix_gate * discrete_repr + (1.0 - mix_gate) * x

        if ust_mode == "shared_only":
            loss_balance = self._code_balance_loss(probs_s)
            loss_orth = torch.tensor(0.0, device=x.device)
        elif ust_mode == "private_only":
            loss_balance = self._code_balance_loss(probs_p)
            loss_orth = torch.tensor(0.0, device=x.device)
        else:
            loss_balance = self._code_balance_loss(probs_s) + self._code_balance_loss(probs_p)
            loss_orth = self._orthogonality_loss(e_s, e_p)
        loss_commit = (1.0 - F.cosine_similarity(discrete_repr, x, dim=-1)).mean()

        return {
            "z_i": z_i,
            "e_s": e_s,
            "e_p": e_p,
            "loss_balance": loss_balance,
            "loss_orth": loss_orth,
            "loss_commit": loss_commit,
            "tokens_s": tokens_s,
            "tokens_p": tokens_p,
            "probs_s": probs_s,
            "probs_p": probs_p,
            "discrete_repr": discrete_repr,
            "continuous_repr": x,
            "tau": tau,
        }

    def forward(self, x, use_ust=True, advance_step=False, ust_mode="full"):
        outputs = self._compute_outputs(
            x,
            use_ust=use_ust,
            advance_step=advance_step,
            ust_mode=ust_mode,
        )
        return (
            outputs["z_i"],
            outputs["e_s"],
            outputs["e_p"],
            outputs["loss_balance"],
            outputs["loss_orth"],
            outputs["loss_commit"],
        )

    def analyze_tokens(self, x, use_ust=True, advance_step=False, ust_mode="full"):
        return self._compute_outputs(
            x,
            use_ust=use_ust,
            advance_step=advance_step,
            ust_mode=ust_mode,
        )

    def compute_distribution_alignment(self, shared_a, shared_b):
        shared_a = F.normalize(shared_a.reshape(-1, shared_a.size(-1)), dim=-1)
        shared_b = F.normalize(shared_b.reshape(-1, shared_b.size(-1)), dim=-1)

        if shared_a.size(0) < 2 or shared_b.size(0) < 2:
            return torch.tensor(0.0, device=shared_a.device)

        mean_loss = (shared_a.mean(dim=0) - shared_b.mean(dim=0)).pow(2).mean()

        centered_a = shared_a - shared_a.mean(dim=0, keepdim=True)
        centered_b = shared_b - shared_b.mean(dim=0, keepdim=True)
        cov_a = centered_a.t().mm(centered_a) / max(shared_a.size(0) - 1, 1)
        cov_b = centered_b.t().mm(centered_b) / max(shared_b.size(0) - 1, 1)
        cov_loss = (cov_a - cov_b).pow(2).mean()

        return mean_loss + cov_loss
