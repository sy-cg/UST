import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_RECBOLE_ROOT = os.path.join(PROJECT_ROOT, "RecBole", "RecBole")


if os.path.isdir(LOCAL_RECBOLE_ROOT) and LOCAL_RECBOLE_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_RECBOLE_ROOT)


def _patch_recbole_lightgcn():
    try:
        import numpy as np
        import scipy.sparse as sp
        import torch
        from recbole.model.general_recommender.lightgcn import LightGCN
    except Exception:
        return

    def _safe_get_norm_adj_mat(self):
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
        return torch.sparse.FloatTensor(
            edge_index, edge_weight, torch.Size(norm_adj.shape)
        )

    LightGCN.get_norm_adj_mat = _safe_get_norm_adj_mat


_patch_recbole_lightgcn()
