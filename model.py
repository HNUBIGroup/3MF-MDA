import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch_geometric.nn import GINConv, GCNConv, HANConv, global_max_pool as gmp


class SPD_CoAttention(nn.Module):
           def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.d_shared = embed_dim // 2
        self.d_private = embed_dim - self.d_shared

        self.proj_d = nn.Linear(embed_dim, embed_dim)
        self.proj_m = nn.Linear(embed_dim, embed_dim)

                self.cross_attn_drug = nn.MultiheadAttention(self.d_shared, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_mirna = nn.MultiheadAttention(self.d_shared, num_heads, dropout=dropout, batch_first=True)

        self.norm_drug = nn.LayerNorm(self.d_shared)
        self.norm_mirna = nn.LayerNorm(self.d_shared)

                self.reconstruct_d = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.reconstruct_m = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, drug_stack, mirna_stack):
        Z_d_proj = self.proj_d(drug_stack)
        Z_m_proj = self.proj_m(mirna_stack)

        P_d, S_d = torch.split(Z_d_proj, [self.d_private, self.d_shared], dim=-1)
        P_m, S_m = torch.split(Z_m_proj, [self.d_private, self.d_shared], dim=-1)

        orth_d = torch.bmm(S_d.transpose(1, 2), P_d)
        orth_m = torch.bmm(S_m.transpose(1, 2), P_m)
        orth_loss = (orth_d ** 2).mean() + (orth_m ** 2).mean()

        S_d_out, _ = self.cross_attn_drug(query=S_d, key=S_m, value=S_m)
        S_d_tilde = self.norm_drug(S_d + S_d_out)

        S_m_out, _ = self.cross_attn_mirna(query=S_m, key=S_d, value=S_d)
        S_m_tilde = self.norm_mirna(S_m + S_m_out)

        drug_recon = torch.cat([P_d, S_d_tilde], dim=-1)
        mirna_recon = torch.cat([P_m, S_m_tilde], dim=-1)

        drug_fused = self.reconstruct_d(drug_recon)
        mirna_fused = self.reconstruct_m(mirna_recon)

        drug_rep = drug_fused.mean(dim=1)
        mirna_rep = mirna_fused.mean(dim=1)

        return drug_rep, mirna_rep, orth_loss


class GIPManifoldEncoder(nn.Module):
    def __init__(self, proj_dim=256, dropout=0.3):
        super(GIPManifoldEncoder, self).__init__()
        prior_tensor = self._load_gip_prior()
        self.register_buffer('gip_prior', prior_tensor)

        in_dim = prior_tensor.shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU()
        )
    def forward(self, node_ids):
        raw_gip = self.gip_prior[node_ids]
        return self.encoder(raw_gip)


class FeatureFusionGate(nn.Module):
    def __init__(self, local_dim=512, struct_dim=256, out_dim=256):
        super(FeatureFusionGate, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(local_dim + struct_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )

    def forward(self, local_feat, struct_feat):
        h = torch.cat([local_feat, struct_feat], dim=1)
        return self.fusion(h)


class PretrainedSeqEncoder(nn.Module):
    def __init__(self, model_name, proj_dim, dropout=0.2, freeze_llm=True):
        super().__init__()
        self.llm = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        llm_out_dim = self.llm.config.hidden_size
        self.fc_proj = nn.Sequential(
            nn.Linear(llm_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            last_hidden = outputs[0]
        else:
            last_hidden = outputs.last_hidden_state
        return self.fc_proj(last_hidden[:, 0, :])


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(
            GINConv(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        )
        self.bns.append(nn.BatchNorm1d(hidden_dim))

     
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, proj_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = self.relu(x)

            x = gmp(x, batch)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MDGraphEncoder(nn.Module):
    def __init__(self, in_dim=256, mid_dim=128, out_dim=256, fc_mid=256, proj_dim=256, dropout=0.2, drug_offset=1578, heads=4):
        super().__init__()
        self.drug_offset = drug_offset
        self.metadata = (['mirna', 'drug'], [('mirna', 'interacts', 'drug'), ('drug', 'rev_interacts', 'mirna')])
        self.han1 = HANConv(in_channels=-1, out_channels=mid_dim, metadata=self.metadata, heads=heads, dropout=dropout)
        self.han2 = HANConv(in_channels=-1, out_channels=out_dim, metadata=self.metadata, heads=heads, dropout=dropout)
        self.fc1 = nn.Linear(out_dim, fc_mid)
        self.fc2 = nn.Linear(fc_mid, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_dict = {'mirna': x[:self.drug_offset], 'drug': x[self.drug_offset:]}
        row, col = edge_index[0], edge_index[1]
        mask_interacts = (row < self.drug_offset) & (col >= self.drug_offset)
        mirna_to_drug = torch.stack([row[mask_interacts], col[mask_interacts] - self.drug_offset], dim=0)
        mask_rev = (row >= self.drug_offset) & (col < self.drug_offset)
        drug_to_mirna = torch.stack([row[mask_rev] - self.drug_offset, col[mask_rev]], dim=0)
        edge_index_dict = {('mirna', 'interacts', 'drug'): mirna_to_drug,
                           ('drug', 'rev_interacts', 'mirna'): drug_to_mirna}

        out_dict = self.han1(x_dict, edge_index_dict)
        out_dict = {k: self.relu(v) for k, v in out_dict.items()}
        out_dict = self.han2(out_dict, edge_index_dict)
        out_dict = {k: self.relu(v) for k, v in out_dict.items()}

        mirna_out, drug_out = out_dict['mirna'], out_dict['drug']
        h = torch.cat([mirna_out, drug_out], dim=0)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class 3MF-MDA(nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_smiles_chars=66,
                 num_drug_node_feats=78, num_mirna_chars=25, proj_dim=256, dropout=0.2, drug_offset=1578,
                 pretrained_embeddings=None, embedding_dim=512):
        super().__init__()
        self.proj_dim = proj_dim
        self.drug_offset = drug_offset
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

             if pretrained_embeddings is not None:
            self.mirna_embedding_table = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32), freeze=True
            )
        else:
            self.mirna_embedding_table = nn.Embedding(1578, embedding_dim)

               self.gip_manifold_encoder = GIPManifoldEncoder(proj_dim=256, dropout=dropout)

             self.mirna_fusion_gate = FeatureFusionGate(local_dim=embedding_dim, struct_dim=256, out_dim=proj_dim)

             self.smiles_llm_encoder = PretrainedSeqEncoder(model_name="seyonec/ChemBERTa-zinc-base-v1", proj_dim=proj_dim, dropout=dropout)
        self.mirna_llm_encoder = PretrainedSeqEncoder(model_name="zhihan1996/DNABERT-2-117M", proj_dim=proj_dim, dropout=dropout)
        self.drug_graph_encoder = GINEncoder(
            in_dim=num_drug_node_feats,
            hidden_dim=256,
            proj_dim=proj_dim,
            num_layers=3,
            dropout=dropout
        )
        self.md_graph_encoder = MDGraphEncoder(in_dim=256, proj_dim=proj_dim, dropout=dropout, drug_offset=drug_offset, heads=4)

        self.spd_co_attention = SPD_CoAttention(embed_dim=proj_dim, num_heads=4, dropout=dropout)

        self.classifier_fc = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output),
            nn.Sigmoid()
        )

    def forward(self, drug_graph_data, bipartite_graph_data, mirna_graph_data=None):
        drug_graph_feat = self.drug_graph_encoder(drug_graph_data.x, drug_graph_data.edge_index, drug_graph_data.batch)
        smiles_data = drug_graph_data.seqdrug.long()
        smiles_data = smiles_data.view(smiles_data.size(0), 2, -1)
        smiles_seq_feat = self.smiles_llm_encoder(smiles_data[:, 0, :], smiles_data[:, 1, :])
        mirna_ids = drug_graph_data.row_indices.long()
        local_attributes = self.mirna_embedding_table(mirna_ids)
        structural_context = self.gip_manifold_encoder(mirna_ids)
        gene_feat = self.mirna_fusion_gate(local_attributes, structural_context)

         mirna_data = drug_graph_data.target.long()
        mirna_data = mirna_data.view(mirna_data.size(0), 2, -1)
        mirna_seq_feat = self.mirna_llm_encoder(mirna_data[:, 0, :], mirna_data[:, 1, :])

        md_node_feats = self.md_graph_encoder(bipartite_graph_data.x, bipartite_graph_data.edge_index)
        rna_md_feat = md_node_feats[drug_graph_data.row_indices, :]
        drug_md_feat = md_node_feats[drug_graph_data.col_indices + self.drug_offset, :]

        drug_feature_stack = torch.stack([smiles_seq_feat, drug_graph_feat, drug_md_feat], dim=1)
        mirna_feature_stack = torch.stack([gene_feat, rna_md_feat, mirna_seq_feat], dim=1)

        drug_emb, mirna_emb, orth_loss = self.spd_co_attention(drug_feature_stack, mirna_feature_stack)
        drug_emb = self.dropout(drug_emb)
        mirna_emb = self.dropout(mirna_emb)

         combined_features = torch.cat([drug_emb, mirna_emb], dim=1)
        output = self.classifier_fc(combined_features)

        return output, orth_loss