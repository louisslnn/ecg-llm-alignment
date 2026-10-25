"""The auxiliary linear head: one logit per example, read off the resampler's output.

From ChatHealthAI's ICU workflow (train_icu_workflow2_softtarget_ce_auprc_two_a100.py):
mean-pool the 64 latents, LayerNorm, one linear unit. Trained by train.py's
--lambda-linear against the binary gold label, and the target its
--lambda-consistency pulls the LLM's own Yes/No logit toward.

One output, not five: the resampler is question-conditioned, so the prefix it pools
already differs per question, and the label is that question's answer.
"""

import torch.nn as nn


class AuxiliaryLinearHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, resampler_output):          # (B, 64, hidden_dim)
        pooled = resampler_output.float().mean(dim=1)
        return self.classifier(self.norm(pooled)).squeeze(-1)
