"""
RandLA-Net — Segmentation sémantique de nuages de points 3D.

Architecture:
  - Encoder: 4 couches de Local Feature Aggregation + Random Sampling (÷4)
  - Decoder: 4 couches de Nearest Neighbor Upsampling + Skip Connections
  - ~1.2M paramètres

Ref: Hu et al., "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds", CVPR 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """Linear → BatchNorm → LeakyReLU (opère sur la dernière dim)."""

    def __init__(self, in_dim, out_dim, bn=True, activation=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(out_dim))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, N, d_in) → (B, N, d_out)"""
        B, N, _ = x.shape
        x = x.reshape(B * N, -1)
        x = self.layers(x)
        return x.reshape(B, N, -1)


class LocalSpatialEncoding(nn.Module):
    """
    Encode la géométrie locale autour de chaque point.
    Pour chaque point i et ses K voisins j:
      r_ij = MLP(p_i || p_j || p_i - p_j || ||p_i - p_j||)
    """

    def __init__(self, d_out):
        super().__init__()
        # Entrée: p_i(3) + p_j(3) + diff(3) + dist(1) = 10
        self.mlp = SharedMLP(10, d_out)

    def forward(self, xyz, neighbor_xyz):
        """
        xyz: (B, N, 3) — positions des points
        neighbor_xyz: (B, N, K, 3) — positions des K voisins
        Returns: (B, N, K, d_out) — encodage spatial
        """
        B, N, K, _ = neighbor_xyz.shape

        # Étendre xyz pour broadcasting: (B, N, 1, 3) → (B, N, K, 3)
        xyz_expanded = xyz.unsqueeze(2).expand(-1, -1, K, -1)

        # Différences et distances
        diff = xyz_expanded - neighbor_xyz  # (B, N, K, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, N, K, 1)

        # Concatenation: [p_i, p_j, p_i-p_j, ||p_i-p_j||]
        encoding = torch.cat([xyz_expanded, neighbor_xyz, diff, dist], dim=-1)
        # (B, N, K, 10)

        # MLP
        encoding = encoding.reshape(B, N * K, 10)
        encoding = self.mlp(encoding)
        encoding = encoding.reshape(B, N, K, -1)

        return encoding


class AttentivePooling(nn.Module):
    """
    Agrégation attention-weighted des features voisins.
    Scores d'attention appris pour pondérer chaque voisin.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(d_in, d_in, bias=False),
            nn.Softmax(dim=-2)  # Softmax sur la dimension K (voisins)
        )
        self.mlp = SharedMLP(d_in, d_out)

    def forward(self, features):
        """
        features: (B, N, K, d_in)
        Returns: (B, N, d_out)
        """
        B, N, K, d = features.shape

        # Attention scores: (B, N, K, d_in) → (B, N, K, d_in)
        scores = self.score_fn(features.reshape(B * N, K, d))
        scores = scores.reshape(B, N, K, d)

        # Weighted sum: (B, N, d_in)
        attended = (features * scores).sum(dim=2)

        # MLP final
        return self.mlp(attended)


class LocalFeatureAggregation(nn.Module):
    """
    Bloc LFA complet = 2x (LocSE + AttPool) + skip connection.
    C'est le building block central de RandLA-Net.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp_in = SharedMLP(d_in, d_out // 2)

        # Branche 1
        self.lse1 = LocalSpatialEncoding(d_out // 2)
        self.pool1 = AttentivePooling(d_out, d_out // 2)

        # Branche 2
        self.lse2 = LocalSpatialEncoding(d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        # Skip connection
        self.shortcut = SharedMLP(d_in, d_out, bn=True, activation=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, xyz, features, neighbor_idx):
        """
        xyz: (B, N, 3)
        features: (B, N, d_in)
        neighbor_idx: (B, N, K)
        Returns: (B, N, d_out)
        """
        B, N, K = neighbor_idx.shape

        # Récupérer les coordonnées des voisins
        neighbor_xyz = self._gather(xyz, neighbor_idx)  # (B, N, K, 3)

        # Première passe
        f = self.mlp_in(features)  # (B, N, d_out//2)
        neighbor_f = self._gather(f, neighbor_idx)  # (B, N, K, d_out//2)

        lse1 = self.lse1(xyz, neighbor_xyz)  # (B, N, K, d_out//2)
        concat1 = torch.cat([neighbor_f, lse1], dim=-1)  # (B, N, K, d_out)
        f = self.pool1(concat1)  # (B, N, d_out//2)

        # Deuxième passe
        neighbor_f2 = self._gather(f, neighbor_idx)
        lse2 = self.lse2(xyz, neighbor_xyz)
        concat2 = torch.cat([neighbor_f2, lse2], dim=-1)
        f = self.pool2(concat2)  # (B, N, d_out)

        # Skip connection
        shortcut = self.shortcut(features)
        return self.lrelu(f + shortcut)

    @staticmethod
    def _gather(x, idx):
        """
        Rassemble les features des voisins.
        x: (B, N, d) → output: (B, N, K, d)
        idx: (B, N, K) — indices dans la dimension N
        """
        B, N, K = idx.shape
        d = x.shape[-1]

        # Clamp indices pour sécurité
        idx_clamped = idx.clamp(0, x.shape[1] - 1)

        # Flatten: (B, N*K) puis expand pour la dim features
        idx_flat = idx_clamped.reshape(B, N * K)
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, d)  # (B, N*K, d)

        # Gather sur dim 1
        gathered = torch.gather(x, 1, idx_expanded)  # (B, N*K, d)

        return gathered.reshape(B, N, K, d)


class RandLANet(nn.Module):
    """
    RandLA-Net complet: Encoder-Decoder pour segmentation sémantique.
    """

    def __init__(self, d_in, num_classes, d_encoder=None, num_layers=4):
        super().__init__()
        if d_encoder is None:
            d_encoder = [32, 64, 128, 256]

        self.num_layers = num_layers

        # Feature lifting initial
        # Run 8: 8→16 — goulot d'étranglement détecté (7 features → 8 dim ≈ aucune capacité repr.)
        # 16 dim permet d'exploiter les features géométriques (linearity, planarity, verticality)
        self.fc_start = SharedMLP(d_in, 16)

        # Encoder
        self.encoders = nn.ModuleList()
        dims = [16] + d_encoder
        for i in range(num_layers):
            self.encoders.append(LocalFeatureAggregation(dims[i], dims[i + 1]))

        # Middle MLP
        self.mlp_middle = SharedMLP(d_encoder[-1], d_encoder[-1])

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = d_encoder[-1] + d_encoder[-1]
            else:
                in_dim = d_encoder[num_layers - i] + d_encoder[num_layers - 1 - i]
            out_dim = d_encoder[num_layers - 1 - i]
            self.decoders.append(SharedMLP(in_dim, out_dim))

        # Classification head
        # Run 8: Dropout(0.5) ajouté pour régularisation (0 params, améliore généralisation)
        self.fc_end1 = SharedMLP(d_encoder[0], 64)
        self.dropout = nn.Dropout(0.5)
        self.fc_end2 = SharedMLP(64, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, xyz, features, batch_data):
        """
        xyz: (B, N, 3)
        features: (B, N, d_in)
        batch_data: dict avec 'neigh_i', 'sub_i', 'up_i' pour i=0..num_layers-1

        Returns: (B, N, num_classes)
        """
        B, N, _ = xyz.shape

        # Feature lifting
        f = self.fc_start(features)  # (B, N, 8)

        # ─── Encoder ───
        encoder_features = []
        current_xyz = xyz
        current_f = f

        for i in range(self.num_layers):
            neigh_idx = batch_data[f"neigh_{i}"]  # (B, N_i, K)

            # LFA
            current_f = self.encoders[i](current_xyz, current_f, neigh_idx)
            encoder_features.append(current_f)

            # Random subsampling
            sub_idx = batch_data[f"sub_{i}"]  # (B, N_i//4)
            current_f = self._subsample(current_f, sub_idx)
            current_xyz = self._subsample(current_xyz, sub_idx)

        # Middle
        current_f = self.mlp_middle(current_f)

        # ─── Decoder ───
        for i in range(self.num_layers):
            enc_level = self.num_layers - 1 - i
            up_idx = batch_data[f"up_{enc_level}"]  # (B, N_enc)

            # Nearest neighbor upsampling
            upsampled = self._upsample(current_f, up_idx)

            # Skip connection
            skip = encoder_features[enc_level]
            concat = torch.cat([upsampled, skip], dim=-1)

            # MLP
            current_f = self.decoders[i](concat)

        # Classification head avec dropout pour régularisation
        out = self.fc_end1(current_f)
        # Dropout appliqué après fc_end1 (en mode training) pour la généralisation cross-scène
        B_o, N_o, d_o = out.shape
        out = self.dropout(out.reshape(B_o * N_o, d_o)).reshape(B_o, N_o, d_o)
        out = self.fc_end2(out)
        B2, N2, d2 = out.shape
        out = self.classifier(out.reshape(B2 * N2, d2))
        out = out.reshape(B2, N2, -1)

        return out

    @staticmethod
    def _subsample(x, idx):
        """
        x: (B, N, d)
        idx: (B, N_sub) — indices des points à garder
        Returns: (B, N_sub, d)
        """
        B, N_sub = idx.shape
        d = x.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
        return torch.gather(x, 1, idx_exp)

    @staticmethod
    def _upsample(x, idx):
        """
        x: (B, N_coarse, d)
        idx: (B, N_fine) — pour chaque point fin, index du voisin coarse
        Returns: (B, N_fine, d)
        """
        B, N_fine = idx.shape
        d = x.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, d)
        return torch.gather(x, 1, idx_exp)


def count_parameters(model):
    """Compte les paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
