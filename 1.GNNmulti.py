# =====================================================================
# MORPH Framework: Deep Geometric Learning for Drug-Protein Interactions
# Module: Hyperparameter Optimization and Graph Representation Pipeline
# =====================================================================

import os
import json
import time
import random
import traceback
import warnings
import re
import gc
from multiprocessing import cpu_count

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
import psutil

warnings.filterwarnings('ignore')


# =====================================================================
# Utilities & Diagnostic Decorators
# =====================================================================

def set_seed(seed=66):
    """Enforces absolute reproducibility across heterogenous backend runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global random seed configured to: {seed}")


def memory_monitor(func):
    """Profiles real-time volatile memory footprints and schedules garbage collection."""

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory

        print(
            f"[METRIC] {func.__name__} execution finalized - Latency: {end_time - start_time:.2f}s, Memory delta: {memory_used:.2f} MB")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    return wrapper


def clean_and_convert_id(id_value):
    """Normalizes raw biological nomenclature strings to ensure matrix token consistency."""
    if pd.isna(id_value):
        return None
    id_str = str(id_value).strip().lower()
    id_str = re.sub(r'[^\w\s]', ' ', id_str)
    id_str = re.sub(r'\s+', ' ', id_str).strip()
    if '.' in id_str:
        try:
            id_str = str(int(float(id_str)))
        except ValueError:
            pass
    return id_str


# =====================================================================
# Hyperparameter and Optimization Architecture Configurations
# =====================================================================

class EnhancedTrainingConfig:
    """Hyperparameter space mapping tailored for Area Under the ROC Curve (AUC) optimization."""

    def __init__(self):
        # Latent Embeddings & Architecture Space
        self.hidden_dim = 512
        self.dropout_rate = 0.2
        self.use_batch_norm = True
        self.num_gcn_layers = 3

        # Multi-Head Attention Parameters
        self.use_attention = True
        self.attention_heads = 8

        # Cross-Modal Fusion Logic
        self.feature_fusion_method = 'attention'

        # Self-Supervised Learning (SSL) Constraints
        self.use_self_supervised = False
        self.ssl_tasks = ['contrastive', 'masking']
        self.ssl_weight = 0.3
        self.masking_ratio = 0.15
        self.temperature = 0.07

        # Optimization & Finetuning Schedules
        self.pretrain_epochs = 8
        self.finetune_epochs = 20
        self.batch_size = 256
        self.learning_rate = 5e-4
        self.weight_decay = 5e-6
        self.grad_clip = 1.0
        self.use_class_weights = True
        self.early_stop_patience = 10

        # Learning Rate Decay Parameters
        self.use_lr_scheduler = True
        self.lr_decay_factor = 0.7
        self.lr_patience = 4
        self.warmup_epochs = 2

        # Robust Cost Criteria Options
        self.use_focal = True
        self.focal_alpha = 0.3
        self.focal_gamma = 1.5
        self.label_smoothing = 0.1

        # Tensor Perturbation & Data Augmentation
        self.use_feature_augmentation = True
        self.augmentation_strength = 0.1

        # IO Configuration
        self.chunk_size = 50000
        self.num_workers = min(4, cpu_count())
        self.test_size = 0.10
        self.val_size = 0.10

        # Precision Accelerations
        self.use_mixed_precision = True
        self.gradient_accumulation_steps = 1


# =====================================================================
# Deep Geometric Learning Interactome Feature Extractor
# =====================================================================

class EnhancedGNNFeatureExtractor(nn.Module):
    """Geometric deep learning block designed for high-resolution drug-protein interactome embedding."""

    def __init__(self, esm2_dim=640, drug_dim=2048, config=None):
        super().__init__()
        self.config = config if config else EnhancedTrainingConfig()
        hidden_dim = self.config.hidden_dim

        # Proteomic Embeddings Transformation Pipeline
        self.protein_preprocessor = nn.Sequential(
            nn.Linear(esm2_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.config.dropout_rate)
        )

        # Molecular Fingerprint Preprocessing Block
        self.drug_preprocessor = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.config.dropout_rate)
        )

        # Graph Neural Network Blocks Installation
        self.gcn_layers = nn.ModuleList()
        for i in range(self.config.num_gcn_layers):
            if i < self.config.num_gcn_layers - 1 and self.config.use_attention:
                self.gcn_layers.append(
                    GATConv(hidden_dim, hidden_dim // max(1, self.config.attention_heads),
                            heads=self.config.attention_heads, dropout=self.config.dropout_rate)
                )
            else:
                self.gcn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Cross-Modal Representation Fusion Layers
        if self.config.feature_fusion_method == 'attention':
            self.feature_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, dropout=self.config.dropout_rate, batch_first=True
            )
            self.feature_fusion = self._attention_fusion
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            )
            self.feature_fusion = self._nonlinear_fusion

        self.output_dim = hidden_dim

        # Self-Supervised Projection Configurations
        if self.config.use_self_supervised:
            if 'contrastive' in self.config.ssl_tasks:
                self.contrastive_projector = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2)
                )
            if 'masking' in self.config.ssl_tasks:
                self.masking_predictor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, esm2_dim)
                )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _attention_fusion(self, features):
        p_base, p_ppi, d_base = features
        feature_seq = torch.stack([p_base, p_ppi, d_base], dim=1)
        attended_features, _ = self.feature_attention(feature_seq, feature_seq, feature_seq)
        return attended_features.mean(dim=1)

    def _nonlinear_fusion(self, features):
        p_base, p_ppi, d_base = features
        return self.fusion_mlp(torch.cat([p_base, p_ppi, d_base], dim=1))

    def contrastive_loss(self, features):
        if not hasattr(self, 'contrastive_projector'):
            return torch.tensor(0.0, device=features.device)
        projected_features = self.contrastive_projector(features)
        projected_features = F.normalize(projected_features, p=2, dim=1)
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(projected_features, projected_features.T) / self.config.temperature
        labels = torch.arange(batch_size).to(features.device)
        loss_i = F.cross_entropy(similarity_matrix, labels)
        loss_j = F.cross_entropy(similarity_matrix.T, labels)
        return (loss_i + loss_j) / 2

    def masking_loss(self, original_features, masked_features, mask_indices):
        if not hasattr(self, 'masking_predictor'):
            return torch.tensor(0.0, device=original_features.device)
        predictions = self.masking_predictor(masked_features)
        return F.mse_loss(predictions[mask_indices], original_features[mask_indices])

    def forward(self, protein_features, drug_features, ppi_data, protein_indices, drug_indices,
                return_features=True, ssl_mode=False, ssl_task=None, mask_indices=None):

        if hasattr(ppi_data, 'protein_ppi_all') and getattr(ppi_data, 'protein_ppi_all', None) is not None:
            protein_ppi_all = ppi_data.protein_ppi_all
            protein_base_all = getattr(ppi_data, 'protein_base_all', None)
            if protein_base_all is None:
                protein_base_all = protein_ppi_all
        else:
            protein_base_all = self.protein_preprocessor(protein_features)
            protein_ppi_all = protein_base_all
            for gcn_layer in self.gcn_layers:
                protein_ppi_all = gcn_layer(protein_ppi_all, ppi_data.edge_index)
                protein_ppi_all = F.leaky_relu(protein_ppi_all, 0.1)

        drug_base_all = self.drug_preprocessor(drug_features)

        batch_protein_base = protein_base_all[protein_indices]
        batch_protein_ppi = protein_ppi_all[protein_indices]
        batch_drug_base = drug_base_all[drug_indices]

        if ssl_mode:
            if ssl_task == 'contrastive':
                return self.contrastive_loss(batch_protein_ppi)
            elif ssl_task == 'masking' and mask_indices is not None:
                return self.masking_loss(protein_features[protein_indices], batch_protein_ppi, mask_indices)
            else:
                return torch.tensor(0.0, device=protein_features.device)

        if return_features:
            return self.feature_fusion((batch_protein_base, batch_protein_ppi, batch_drug_base))
        return batch_protein_base, batch_protein_ppi, batch_drug_base


# =====================================================================
# Memory-Optimized Tensor Dataset
# =====================================================================

class OptimizedMemoryGraphDataset(Dataset):
    """Custom index mapping tensor map for virtual screening applications."""

    def __init__(self, labels, protein_to_idx, drug_to_idx, graph_data):
        self.labels = labels
        self.protein_to_idx = protein_to_idx
        self.drug_to_idx = drug_to_idx
        self.graph_data = graph_data
        self.valid_pairs = []
        self.labels_list = []

        print("[INFO] Constructing vectorized graph pair mapping table...")
        for (drug_id, protein_id), label in tqdm(labels.items(), desc="Processing target labels"):
            if protein_id in protein_to_idx and drug_id in drug_to_idx:
                self.valid_pairs.append((drug_id, protein_id))
                self.labels_list.append(label)

        self.labels_tensor = torch.LongTensor(self.labels_list)
        print(f"[INFO] Optimized dataset initialized with total pairs: {len(self.valid_pairs)}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        drug_id, protein_id = self.valid_pairs[idx]
        return self.protein_to_idx[protein_id], self.drug_to_idx[drug_id], self.labels_tensor[idx]


# =====================================================================
# Data Ingestion & Preprocessing Routines
# =====================================================================

@memory_monitor
def optimized_load_data(esm2_file, drug_fp_file, y_file, ppi_file, sample_ratio=1.0, chunk_size=100000):
    """Ingests high-dimensional molecular descriptors and structures token mappings."""
    print("[INFO] Loading high-dimensional representations...")
    y_dtypes = {'ChemicalName': 'category', 'GeneSymbol': 'category', 'Relation': 'int'}

    y_df = pd.read_csv(y_file, dtype=y_dtypes)
    y_df['ChemicalName'] = y_df['ChemicalName'].astype(str).apply(clean_and_convert_id)
    y_df['GeneSymbol'] = y_df['GeneSymbol'].astype(str).apply(clean_and_convert_id)
    y_df = y_df.dropna(subset=['ChemicalName', 'GeneSymbol'])

    if sample_ratio < 1.0:
        original_size = len(y_df)
        y_df = y_df.sample(frac=sample_ratio, random_state=42)
        print(f"[INFO] Dataset downsampled fraction applied: {original_size} -> {len(y_df)}")

    required_proteins = set(y_df['GeneSymbol'].unique())
    required_drugs = set(y_df['ChemicalName'].unique())

    print(
        f"[INFO] Distinct active spaces: {len(required_proteins)} proteins, {len(required_drugs)} chemicals detected.")

    print("[INFO] Reading proteomic language embeddings chunk stream...")
    protein_esm2_features = {}
    for chunk in pd.read_csv(esm2_file, chunksize=chunk_size):
        chunk['GeneSymbol'] = chunk['GeneSymbol'].astype(str).apply(clean_and_convert_id)
        chunk = chunk.dropna(subset=['GeneSymbol'])[chunk['GeneSymbol'].isin(required_proteins)]
        embedding_cols = [col for col in chunk.columns if col.startswith('Embedding_')]
        for protein_id, features in zip(chunk['GeneSymbol'], chunk[embedding_cols].values):
            protein_esm2_features[protein_id] = features.astype(np.float32)

    print("[INFO] Reading small-molecule chemical fingerprint chunk stream...")
    drug_fp_features = {}
    for chunk in pd.read_csv(drug_fp_file, chunksize=chunk_size):
        chunk['ChemicalName'] = chunk['ChemicalName'].astype(str).apply(clean_and_convert_id)
        chunk = chunk.dropna(subset=['ChemicalName'])[chunk['ChemicalName'].isin(required_drugs)]
        fp_cols = [col for col in chunk.columns if col.startswith('FP_')]
        for drug_id, features in zip(chunk['ChemicalName'], chunk[fp_cols].values):
            drug_fp_features[drug_id] = features.astype(np.float32)

    labels = {}
    for _, row in y_df.iterrows():
        drug_id = row['ChemicalName']
        protein_id = row['GeneSymbol']
        if drug_id in drug_fp_features and protein_id in protein_esm2_features:
            labels[(drug_id, protein_id)] = int(row['Relation'])

    print(f"[SUCCESS] Ingested coordinates map count: {len(labels)}")
    print(
        f"[METRIC] Global balance distributions: 0={list(labels.values()).count(0)}, 1={list(labels.values()).count(1)}")

    print("[INFO] Setting up background protein interactome adjacency maps...")
    ppi_dtypes = {'protein1_Gene': 'category', 'protein2_Gene': 'category'}
    ppi_df = pd.read_csv(ppi_file, dtype=ppi_dtypes)
    ppi_df['protein1_Gene'] = ppi_df['protein1_Gene'].astype(str).apply(clean_and_convert_id)
    ppi_df['protein2_Gene'] = ppi_df['protein2_Gene'].astype(str).apply(clean_and_convert_id)
    ppi_df = ppi_df.dropna(subset=['protein1_Gene', 'protein2_Gene'])

    return protein_esm2_features, drug_fp_features, labels, ppi_df


@memory_monitor
def build_graph_data(protein_esm2_features, drug_fp_features, labels, ppi_df, hidden_dim=512):
    """Transforms isolated sequence arrays into uniform geometric graph data configurations."""
    print("[INFO] Building network topology matrices...")

    all_drugs_in_y = set(drug_id for drug_id, _ in labels.keys())
    all_proteins_in_y = set(protein_id for _, protein_id in labels.keys())

    all_drugs = [drug for drug in all_drugs_in_y if drug in drug_fp_features]
    all_proteins = [protein for protein in all_proteins_in_y if protein in protein_esm2_features]

    print(f"[INFO] Topology dimensions: {len(all_proteins)} nodes (proteins), {len(all_drugs)} instances (drugs)")

    protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
    drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}

    valid_labels = {(d, p): l for (d, p), l in labels.items() if d in drug_to_idx and p in protein_to_idx}

    ppi_edges = []
    for _, row in ppi_df.iterrows():
        p1, p2 = str(row['protein1_Gene']), str(row['protein2_Gene'])
        if p1 in protein_to_idx and p2 in protein_to_idx:
            src, dst = protein_to_idx[p1], protein_to_idx[p2]
            ppi_edges.extend([[src, dst], [dst, src]])

    ppi_edge_index = torch.tensor(ppi_edges, dtype=torch.long).t().contiguous() if ppi_edges else torch.empty((2, 0),
                                                                                                              dtype=torch.long)
    print(f"[INFO] Successfully configured total edge index dimensions: {ppi_edge_index.shape[1]}")

    protein_esm2_matrix = torch.stack([torch.FloatTensor(protein_esm2_features[p]) for p in all_proteins])
    drug_fp_matrix = torch.stack([torch.FloatTensor(drug_fp_features[d]) for d in all_drugs])
    ppi_data = Data(x=torch.zeros((len(all_proteins), hidden_dim), dtype=torch.float32), edge_index=ppi_edge_index)

    return {
        'protein_esm2_matrix': protein_esm2_matrix,
        'drug_fp_matrix': drug_fp_matrix,
        'protein_to_idx': protein_to_idx,
        'drug_to_idx': drug_to_idx,
        'all_proteins': all_proteins,
        'all_drugs': all_drugs,
        'ppi_data': ppi_data,
        'labels': valid_labels
    }


def normalize_features(graph_data):
    """Applies global Z-score standardization parameters onto target representation spaces."""
    graph_data['protein_esm2_matrix_raw'] = graph_data['protein_esm2_matrix'].clone()
    graph_data['drug_fp_matrix_raw'] = graph_data['drug_fp_matrix'].clone()

    esm2_mean, esm2_std = graph_data['protein_esm2_matrix'].mean(dim=0), graph_data['protein_esm2_matrix'].std(dim=0)
    fp_mean, fp_std = graph_data['drug_fp_matrix'].mean(dim=0), graph_data['drug_fp_matrix'].std(dim=0)

    graph_data.update({'esm2_mean': esm2_mean, 'esm2_std': esm2_std, 'fp_mean': fp_mean, 'fp_std': fp_std})
    graph_data['protein_esm2_matrix'] = (graph_data['protein_esm2_matrix'] - esm2_mean) / (esm2_std + 1e-8)
    graph_data['drug_fp_matrix'] = (graph_data['drug_fp_matrix'] - fp_mean) / (fp_std + 1e-8)

    print("[SUCCESS] Feature standardizations calculated.")
    return graph_data


def strict_data_split(labels, test_size=0.2, val_size=0.1, random_state=42):
    """Enforces rigorous node-disjoint partitioning to preclude latent information leakage."""
    print("[INFO] Initializing leak-proof validation data splitting strategy...")

    all_drugs = list(set(drug_id for drug_id, _ in labels.keys()))
    all_proteins = list(set(protein_id for _, protein_id in labels.keys()))

    np.random.seed(random_state)
    test_drugs = set(np.random.choice(all_drugs, int(len(all_drugs) * test_size), replace=False))
    test_proteins = set(np.random.choice(all_proteins, int(len(all_proteins) * test_size), replace=False))

    test_pairs, train_val_pairs = [], []
    for pair in labels.keys():
        if pair[0] in test_drugs or pair[1] in test_proteins:
            test_pairs.append(pair)
        else:
            train_val_pairs.append(pair)

    val_size_relative = val_size / (1 - test_size)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs, test_size=val_size_relative, random_state=random_state,
        stratify=[labels[p] for p in train_val_pairs]
    )

    print(
        f"[METRIC] Training array: {len(train_pairs)} | Validation array: {len(val_pairs)} | Test array: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs


# =====================================================================
# Robust Optimization Loss Criteria
# =====================================================================

class FocalLoss(nn.Module):
    """Focal Loss implementation targeting heavy class imbalances during screening."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (self.alpha * (1 - pt) ** self.gamma * ce)
        if self.reduction == 'mean': return focal.mean()
        if self.reduction == 'sum': return focal.sum()
        return focal


# =====================================================================
# Supervised End-to-End Fine-Tuning Protocol
# =====================================================================

class EnhancedFeatureExtractorTrainer:
    """Supervised framework orchestrator equipped with multi-objective tracking capacities."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.classifier = None
        self.best_val_auc = 0.0
        self.best_model_state = None
        self.best_classifier_state = None
        self.scaler = torch.cuda.amp.GradScaler() if (
                    self.config.use_mixed_precision and torch.cuda.is_available()) else None
        self.focal_loss_fn = FocalLoss(alpha=self.config.focal_alpha,
                                       gamma=self.config.focal_gamma) if self.config.use_focal else None
        self.scheduler = None
        self.last_lr = None

    def precompute_ppi_embeddings(self, model, graph_data):
        """Pre-computes interactome embeddings across the background graph structure."""
        print("[INFO] Caching static graph network embeddings...")
        model.eval()
        with torch.no_grad():
            protein_feats = graph_data['protein_esm2_matrix'].to(self.device, non_blocking=True)
            protein_base_all = model.protein_preprocessor(protein_feats)
            edge_index = graph_data['ppi_data'].edge_index.to(self.device, non_blocking=True)

            protein_ppi_all = protein_base_all
            ppi_tmp = Data(edge_index=edge_index)
            for g in model.gcn_layers:
                protein_ppi_all = g(protein_ppi_all, ppi_tmp.edge_index)
                protein_ppi_all = F.leaky_relu(protein_ppi_all, 0.1)

            graph_data['ppi_data'].protein_base_all = protein_base_all.detach()
            graph_data['ppi_data'].protein_ppi_all = protein_ppi_all.detach()
            graph_data['drug_fp_matrix'] = graph_data['drug_fp_matrix'].to(self.device, non_blocking=True)
            graph_data['protein_esm2_matrix'] = graph_data['protein_esm2_matrix'].to(self.device, non_blocking=True)
        print("[SUCCESS] Static topological cache matrix operational.")

    def _create_masking_indices(self, batch_size, feature_dim):
        num_masks = int(batch_size * self.config.masking_ratio)
        return torch.randperm(batch_size)[:num_masks] if num_masks > 0 else None

    def train(self, graph_data, labels):
        print("[INFO] Launching deep feature extraction initialization sequence...")
        train_pairs, val_pairs, _ = strict_data_split(labels, test_size=self.config.test_size,
                                                      val_size=self.config.val_size)

        dataset = OptimizedMemoryGraphDataset(labels, graph_data['protein_to_idx'], graph_data['drug_to_idx'],
                                              graph_data)
        train_pairs_set, val_pairs_set = set(train_pairs), set(val_pairs)

        train_indices = [i for i, pair in enumerate(dataset.valid_pairs) if tuple(pair) in train_pairs_set]
        val_indices = [i for i, pair in enumerate(dataset.valid_pairs) if tuple(pair) in val_pairs_set]

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=self.config.batch_size,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=self.config.batch_size * 2,
                                shuffle=False, pin_memory=True)

        self.model = EnhancedGNNFeatureExtractor(config=self.config).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.output_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)
        ).to(self.device)

        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.classifier.parameters()),
                                      lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        graph_data['protein_esm2_matrix'] = graph_data['protein_esm2_matrix'].to(self.device, non_blocking=True)
        graph_data['drug_fp_matrix'] = graph_data['drug_fp_matrix'].to(self.device, non_blocking=True)
        graph_data['ppi_data'].edge_index = graph_data['ppi_data'].edge_index.to(self.device, non_blocking=True)

        self.precompute_ppi_embeddings(self.model, graph_data)

        if self.config.use_self_supervised:
            self._self_supervised_pretrain(self.model, train_loader, graph_data, optimizer)

        self._supervised_finetune(self.model, train_loader, val_loader, graph_data, optimizer)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.classifier.load_state_dict(self.best_classifier_state)
            print(
                f"[SUCCESS] Global optimized target checkpoint retrieved. Top Validation AUC: {self.best_val_auc:.4f}")

        return self.model

    def _self_supervised_pretrain(self, model, train_loader, graph_data, optimizer):
        print("[INFO] Commencing self-supervised pre-training protocol...")
        model.train()
        for epoch in range(self.config.pretrain_epochs):
            total_loss, batch_count = 0, 0
            for protein_indices, drug_indices, _ in tqdm(train_loader, desc=f'SSL Pretrain Epoch {epoch + 1}'):
                protein_indices, drug_indices = protein_indices.to(self.device), drug_indices.to(self.device)
                optimizer.zero_grad()
                ssl_task = random.choice(self.config.ssl_tasks)
                ssl_loss = 0
                if ssl_task == 'contrastive':
                    ssl_loss = model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                     graph_data['ppi_data'], protein_indices, drug_indices, ssl_mode=True,
                                     ssl_task=ssl_task)
                elif ssl_task == 'masking':
                    mask_indices = self._create_masking_indices(len(protein_indices),
                                                                graph_data['protein_esm2_matrix'].shape[1])
                    if mask_indices is not None:
                        ssl_loss = model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                         graph_data['ppi_data'], protein_indices, drug_indices, ssl_mode=True,
                                         ssl_task=ssl_task, mask_indices=mask_indices)
                if ssl_loss != 0:
                    ssl_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    total_loss += ssl_loss.item()
                    batch_count += 1
            if batch_count > 0:
                print(
                    f'[INFO] Pretrain Epoch {epoch + 1} Mean Objective Criterion Value: {total_loss / batch_count:.4f}')

    def _supervised_finetune(self, model, train_loader, val_loader, graph_data, optimizer):
        print("[INFO] Commencing supervised execution and hyperparameter mapping...")
        if self.config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                        factor=self.config.lr_decay_factor,
                                                                        patience=self.config.lr_patience)

        self.last_lr = self.config.learning_rate
        criterion = self.focal_loss_fn if self.config.use_focal else nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing)

        patience_counter, best_val_auc = 0, 0

        for epoch in range(self.config.finetune_epochs):
            if epoch < self.config.warmup_epochs:
                current_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in optimizer.param_groups: param_group['lr'] = current_lr

            model.train()
            self.classifier.train()
            train_loss, all_preds, all_labels, batch_count = 0, [], [], 0

            for protein_indices, drug_indices, labels_batch in tqdm(train_loader, desc=f'Finetune Epoch {epoch + 1}'):
                protein_indices, drug_indices, labels_batch = protein_indices.to(self.device), drug_indices.to(
                    self.device), labels_batch.to(self.device)
                optimizer.zero_grad()

                features = model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                 graph_data['ppi_data'], protein_indices, drug_indices, return_features=True)
                if self.config.use_feature_augmentation and random.random() > 0.5:
                    features = features + torch.randn_like(features) * self.config.augmentation_strength

                outputs = self.classifier(features)
                supervised_loss = criterion(outputs, labels_batch)

                if self.config.use_self_supervised and self.config.ssl_weight > 0:
                    ssl_loss = model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                     graph_data['ppi_data'], protein_indices, drug_indices, ssl_mode=True,
                                     ssl_task=random.choice(self.config.ssl_tasks))
                    total_loss = supervised_loss + self.config.ssl_weight * ssl_loss
                else:
                    total_loss = supervised_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(self.classifier.parameters()),
                                               self.config.grad_clip)
                optimizer.step()

                train_loss += total_loss.item()
                all_preds.extend(F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                batch_count += 1

            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
            val_auc = self._evaluate_model(model, val_loader, graph_data)

            if self.config.use_lr_scheduler and self.scheduler is not None:
                self.scheduler.step(val_auc)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != self.last_lr:
                    print(f"[INFO] Learning rate adjusted down from {self.last_lr:.2e} to {new_lr:.2e}")
                    self.last_lr = new_lr

            print(
                f'[METRIC] Epoch {epoch + 1} -> Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_val_auc = val_auc
                self.best_model_state = model.state_dict().copy()
                self.best_classifier_state = self.classifier.state_dict().copy()
                patience_counter = 0
                print(f'  --> [UPDATE] Target optimized weights cataloged. Val AUC: {val_auc:.4f}')
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stop_patience:
                print(
                    f"[INFO] Early stopping constraint triggered at epoch {epoch + 1}. Terminal Val AUC: {best_val_auc:.4f}")
                break

    def _evaluate_model(self, model, data_loader, graph_data):
        model.eval()
        self.classifier.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for protein_indices, drug_indices, labels_batch in data_loader:
                protein_indices, drug_indices = protein_indices.to(self.device), drug_indices.to(self.device)
                features = model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                 graph_data['ppi_data'], protein_indices, drug_indices, return_features=True)
                probs = F.softmax(self.classifier(features), dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels_batch.numpy())

        return roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5


# =====================================================================
# Latent Embeddings Extraction Protocol
# =====================================================================

class FeatureExtractor:
    """Extracts downstream dense feature vectors from hidden topologies."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def extract_features(self, graph_data, dataset):
        print("[INFO] Launching downstream feature representation extractions...")
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)

        all_features, all_labels, all_pairs = [], [], []
        with torch.no_grad():
            for protein_indices, drug_indices, labels_batch in tqdm(dataloader, desc="Extracting vectors"):
                protein_indices, drug_indices = protein_indices.to(self.device), drug_indices.to(self.device)
                features = self.model(graph_data['protein_esm2_matrix'], graph_data['drug_fp_matrix'],
                                      graph_data['ppi_data'], protein_indices, drug_indices, return_features=True)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels_batch.numpy())
                all_pairs.extend([dataset.valid_pairs[i] for i in range(len(labels_batch))])

        features_array = np.vstack(all_features)
        print(f"[SUCCESS] Latent embeddings successfully exported. Target shape: {features_array.shape}")
        return {'features': features_array, 'labels': np.concatenate(all_labels), 'pairs': all_pairs,
                'feature_dim': features_array.shape[1]}


def save_graph_data(graph_data, output_path):
    """Serializes independent topological graph components onto persistent space."""
    print("[INFO] Serializing calculated target graph instances...")
    graph_data_cpu = {}
    for key, value in graph_data.items():
        graph_data_cpu[key] = value.cpu() if torch.is_tensor(value) else value

    if 'ppi_data' in graph_data_cpu:
        ppi_data = graph_data_cpu['ppi_data']
        if hasattr(ppi_data, 'x') and ppi_data.x is not None: ppi_data.x = ppi_data.x.cpu()
        if hasattr(ppi_data, 'edge_index'): ppi_data.edge_index = ppi_data.edge_index.cpu()
        if hasattr(ppi_data, 'protein_base_all'): ppi_data.protein_base_all = ppi_data.protein_base_all.cpu()
        if hasattr(ppi_data, 'protein_ppi_all'): ppi_data.protein_ppi_all = ppi_data.protein_ppi_all.cpu()

    with open(output_path, 'wb') as f:
        pickle.dump(graph_data_cpu, f)
    print(f"[SUCCESS] Interconnected systems cataloged at path: {output_path}")


# =====================================================================
# Optuna Automated Hyperparameter Optimization Core
# =====================================================================

def create_tunable_config(trial):
    """Defines search boundary sweeps across hyperparameter variants."""
    cfg = EnhancedTrainingConfig()
    cfg.hidden_dim = trial.suggest_categorical('hidden_dim', [256, 384, 512, 768])
    cfg.dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4)
    cfg.num_gcn_layers = trial.suggest_int('num_gcn_layers', 2, 4)
    cfg.attention_heads = trial.suggest_categorical('attention_heads', [4, 8])
    cfg.learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
    cfg.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    cfg.focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
    cfg.focal_gamma = trial.suggest_float('focal_gamma', 0.8, 3.0)
    cfg.augmentation_strength = trial.suggest_float('augmentation_strength', 0.0, 0.25)
    cfg.lr_decay_factor = trial.suggest_float('lr_decay_factor', 0.5, 0.9)
    cfg.finetune_epochs = trial.suggest_int('finetune_epochs', 4, 20)
    cfg.pretrain_epochs = 0
    cfg.use_self_supervised = False
    cfg.use_feature_augmentation = True
    return cfg


def objective(trial, device, protein_esm2_features, drug_fp_features, labels, ppi_df):
    config = create_tunable_config(trial)
    graph_data = build_graph_data(protein_esm2_features, drug_fp_features, labels, ppi_df, hidden_dim=config.hidden_dim)
    graph_data = normalize_features(graph_data)
    trainer = EnhancedFeatureExtractorTrainer(config, device)
    try:
        _ = trainer.train(graph_data, graph_data['labels'])
        val_auc = trainer.best_val_auc
    except Exception:
        print('[ERROR] Target trial iteration encountered processing failure anomalies.')
        traceback.print_exc()
        val_auc = 0.0
    trial.report(val_auc, 0)
    return val_auc


def run_hyperparam_search(n_trials, device, protein_esm2_features, drug_fp_features, labels, ppi_df,
                          study_name='gnn_opt_study', output_dir='optuna_results'):
    os.makedirs(output_dir, exist_ok=True)
    study = optuna.create_study(direction='maximize', study_name=study_name, sampler=optuna.samplers.TPESampler())
    func = lambda trial: objective(trial, device, protein_esm2_features, drug_fp_features, labels, ppi_df)
    study.optimize(func, n_trials=n_trials)

    best, best_val = study.best_params, study.best_value
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump({'best_params': best, 'best_val_auc': best_val}, f, indent=2)

    try:
        import joblib
        joblib.dump(study, os.path.join(output_dir, 'optuna_study.pkl'))
    except Exception:
        pass

    print('\n[OPTUNA] Hyperparameter search sequence terminated.')
    print(f'[OPTUNA] Optimized Target Peak AUC: {best_val:.4f}')
    print(f'[OPTUNA] Optimized Parameter Configuration Array: {best}')
    return study


# =====================================================================
# Main Pipeline Interface Orchestration
# =====================================================================

@memory_monitor
def enhanced_main(run_optuna=False, n_trials=20, sample_ratio=1.0):
    print('=' * 70)
    print('MORPH: Deep Geometric Learning Pipeline - Convergence Protocol (Optuna Integrated)')
    print('=' * 70)

    config = EnhancedTrainingConfig()
    output_dir = 'enhanced_features'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('[INFO] Executing under Apple Metal Performance Shaders (MPS) environments.')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('[INFO] Executing under unified CUDA acceleration architectures.')
    else:
        device = torch.device('cpu')
        print('[INFO] Fallback configured onto CPU environments.')
        config.use_mixed_precision = False

    set_seed(66)

    esm2_path = './Rawdata/protein_esm2_features_final_new.csv'
    drug_fp_path = './Rawdata/drug_fp_features_final_new.csv'
    y_path = './Rawdata/Y_clean_no_conflict_noBinding_ProtExp.csv'
    ppi_path = './Rawdata/PPI.v12.20260122.csv'

    print('\n[STAGE 1] Launching dynamic vector file parsing allocations...')
    protein_esm2_features, drug_fp_features, labels, ppi_df = optimized_load_data(
        esm2_path, drug_fp_path, y_path, ppi_path, sample_ratio=sample_ratio, chunk_size=config.chunk_size
    )

    if run_optuna:
        print('\n[STAGE 2] Enforcing automated hyperparameter search loops via Optuna APIs...')
        return run_hyperparam_search(n_trials=n_trials, device=device, protein_esm2_features=protein_esm2_features,
                                     drug_fp_features=drug_fp_features, labels=labels, ppi_df=ppi_df)

    print('\n[STAGE 2] Mapping multi-modal elements into uniform graphs...')
    graph_data = build_graph_data(protein_esm2_features, drug_fp_features, labels, ppi_df, config.hidden_dim)
    labels = graph_data['labels']
    graph_data = normalize_features(graph_data)

    print('\n[STAGE 3] Training initialized feature extractor architectures...')
    trainer = EnhancedFeatureExtractorTrainer(config, device)
    trained_model = trainer.train(graph_data, labels)

    print('\n[STAGE 4] Querying structural vectors from optimized spaces...')
    dataset = OptimizedMemoryGraphDataset(labels, graph_data['protein_to_idx'], graph_data['drug_to_idx'], graph_data)
    extractor = FeatureExtractor(trained_model, device)
    features_dict = extractor.extract_features(graph_data, dataset)

    print('\n[STAGE 5] Exporting target tracking configurations...')
    os.makedirs(output_dir, exist_ok=True)
    save_graph_data(graph_data, f'{output_dir}/graph_data.pkl')

    with open(f'{output_dir}/enhanced_features.pkl', 'wb') as f:
        pickle.dump(features_dict, f)

    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config.__dict__,
        'best_val_auc': trainer.best_val_auc
    }, f'{output_dir}/enhanced_feature_extractor_model.pth')

    print('\n' + '=' * 70)
    print('[SUCCESS] Deep virtual screening pipeline executed successfully.')
    print('=' * 70)
    print(f"[METRIC] Total dimensional vectors compiled: {features_dict['features'].shape[1]}")
    print(f"[METRIC] Total verified coordinates maps handled: {features_dict['features'].shape[0]}")
    print(f"[METRIC] Optimized validation target AUC verified: {trainer.best_val_auc:.4f}")
    return features_dict, trained_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--optuna', action='store_true', help='Execute multi-objective automated optimization routines')
    parser.add_argument('--n_trials', type=int, default=20, help='Total parameter boundary search sweeps to execute')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Proportion to slice from coordinate tables')
    args = parser.parse_args()

    if args.optuna:
        enhanced_main(run_optuna=True, n_trials=args.n_trials, sample_ratio=args.sample_ratio)
    else:
        enhanced_main(run_optuna=False, n_trials=0, sample_ratio=args.sample_ratio)