"""BrainSAIL model architecture.

Implements the BrainSAIL classification model, which consists of three
sequential components:

    1. Visual Adapter    — lightweight domain adaptation on frozen features
    2. Soft Attention MIL — per-class weighted aggregation of MRI slices
    3. Cosine Classifier  — similarity-based classification against text anchors

Example:
    >>> anchors = torch.randn(5, 768)
    >>> model = BrainSAIL(num_classes=5, feat_dim=768, text_anchors=anchors)
    >>> bag = torch.randn(26, 768)   # 26 slices, 768-dim features
    >>> logits, loss = model(bag)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainSAIL(nn.Module):
    """Few-shot brain MRI classifier using Soft Attention MIL and text anchors.

    Processes a bag of MRI slice features per patient and produces
    multi-label predictions by comparing aggregated visual representations
    against class-level text anchor embeddings via cosine similarity.

    Args:
        num_classes (int): Number of output classes. Defaults to 5.
        feat_dim (int): Dimension of input feature vectors. Defaults to 768.
        text_anchors (torch.Tensor, optional): Pre-built text anchor embeddings
            of shape ``[num_classes, feat_dim]``. If None, anchors are randomly
            initialized. Defaults to None.
        dropout (float): Dropout probability applied after the adapter and
            within attention scorers. Defaults to 0.3.
    """

    def __init__(
        self,
        num_classes: int = 5,
        feat_dim: int = 768,
        text_anchors: torch.Tensor | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # --- Text anchors (one per class) ---
        # Initialized from pre-built LLM embeddings when available;
        # otherwise randomly initialized on the unit hypersphere.
        if text_anchors is not None:
            self.text_anchors = nn.Parameter(text_anchors.float())
        else:
            self.text_anchors = nn.Parameter(
                F.normalize(torch.randn(num_classes, feat_dim), dim=-1)
            )

        # --- Visual Adapter ---
        # Bottleneck MLP (768 → 192 → 768) with residual connection.
        # Output layer is zero-initialized so the adapter acts as an identity
        # function at the start of training (safe residual property).
        self.adapter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 4, feat_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        # --- Soft Attention MIL scorers ---
        # One independent scorer per class so each pathology can attend
        # to different slices (e.g., WMH → periventricular, Atrophy → ventricles).
        self.attn_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 8),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 8, 1),
            )
            for _ in range(num_classes)
        ])
        for scorer in self.attn_scorers:
            nn.init.normal_(scorer[0].weight, std=0.01)
            nn.init.normal_(scorer[-1].weight, std=0.01)

        # --- Cosine classifier parameters ---
        # Learnable temperature controls attention sharpness.
        # Logit scale and per-class bias mirror the CLIP convention.
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.logit_scale = nn.Parameter(torch.tensor(14.0))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor | None = None,
        epoch: int = 0,
        pos_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single patient bag.

        Args:
            x (torch.Tensor): Slice feature tensor of shape ``[S, D]`` or
                ``[1, S, D]``, where S is the number of slices and D is
                ``feat_dim``.
            label (torch.Tensor, optional): Ground-truth multi-label vector
                of shape ``[num_classes]`` or ``[1, num_classes]``. Required
                for loss computation. Defaults to None.
            epoch (int): Current training epoch (reserved for future schedulers).
                Defaults to 0.
            pos_weight (torch.Tensor, optional): Per-class positive weights for
                ``BCEWithLogitsLoss``. Shape ``[num_classes]``. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - **logits** — raw scores of shape ``[1, num_classes]``.
                - **loss** — scalar tensor; ``0.0`` when ``label`` is None.
        """
        # Normalize input shape to [S, D]
        if x.dim() == 3:
            x = x.squeeze(0)
        x = x.float()

        # 1. Visual Adapter with residual connection
        x = x + self.adapter(x)
        x = self.dropout(x)

        # 2. Soft Attention MIL — per-class bag-level representation
        bag_reprs = []
        for c in range(self.num_classes):
            scores = self.attn_scorers[c](x) * self.temperature  # [S, 1]
            weights = F.softmax(scores, dim=0)                    # [S, 1]
            bag_reprs.append((x * weights).sum(dim=0))            # [D]
        bag_reprs = torch.stack(bag_reprs)  # [C, D]

        # 3. Cosine Classifier — compare bag repr to text anchors
        bag_norm = F.normalize(bag_reprs, p=2, dim=-1)
        txt_norm = F.normalize(self.text_anchors, p=2, dim=-1)
        logits = (bag_norm * txt_norm).sum(dim=-1) * self.logit_scale + self.bias
        logits = logits.unsqueeze(0)  # [1, C]

        # 4. Loss computation (training only)
        loss = torch.tensor(0.0, device=x.device)
        if label is not None:
            label = label.float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            # Label smoothing: pushes targets away from hard 0/1
            smoothed = label * 0.95 + 0.025
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, smoothed, pos_weight=pos_weight
            )

            # Orthogonality loss: penalizes similar text anchors across classes
            gram = txt_norm @ txt_norm.T
            eye = torch.eye(self.num_classes, device=x.device)
            orth_loss = ((gram - eye) ** 2).mean()

            loss = cls_loss + 0.05 * orth_loss

        return logits, loss

    def build_optimizer(
        self, lr_base: float = 1e-4, weight_decay: float = 1e-2
    ) -> torch.optim.Optimizer:
        """Build AdamW optimizer with per-group learning rates.

        Text anchors use a 3× higher learning rate to adapt quickly from
        general semantic space to the brain MRI domain. The adapter uses a
        0.5× rate to preserve pre-trained UniMedCLIP representations.

        Args:
            lr_base (float): Base learning rate. Defaults to ``1e-4``.
            weight_decay (float): AdamW weight decay. Defaults to ``1e-2``.

        Returns:
            torch.optim.AdamW: Configured optimizer.
        """
        return torch.optim.AdamW(
            [
                {"params": [self.text_anchors],                                        "lr": lr_base * 3.0},
                {"params": self.adapter.parameters(),                                  "lr": lr_base * 0.5},
                {"params": list(self.attn_scorers.parameters()) + [self.temperature],  "lr": lr_base},
                {"params": [self.logit_scale, self.bias],                              "lr": lr_base},
            ],
            weight_decay=weight_decay,
        )


def build_model(
    text_anchors: torch.Tensor | None = None,
    num_classes: int = 5,
    feat_dim: int = 768,
    dropout: float = 0.3,
) -> BrainSAIL:
    """Convenience factory for BrainSAIL.

    Args:
        text_anchors (torch.Tensor, optional): Pre-built anchor embeddings.
        num_classes (int): Number of pathology classes. Defaults to 5.
        feat_dim (int): Feature dimension. Defaults to 768.
        dropout (float): Dropout probability. Defaults to 0.3.

    Returns:
        BrainSAIL: Instantiated model.
    """
    return BrainSAIL(
        num_classes=num_classes,
        feat_dim=feat_dim,
        text_anchors=text_anchors,
        dropout=dropout,
    )
