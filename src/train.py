"""Training the alignment modules by distillation from a teacher LLM.

SETUP (per Bo Hong Wang's suggestion):
  1. Teacher (gpt-oss 120B) receives the GROUND-TRUTH ECG labels and writes a
     summary. This is the target.
  2. Student receives only the ECG embedding plus a generic question
     ("Anything wrong with this ECG?") and must produce the same summary.
  3. Loss: next-token cross-entropy over the target summary.
  4. Gradients flow BACK THROUGH the frozen LLM into the resampler and
     projection. The LLM's own weights never update, but it still passes
     gradients, which is what lets the resampler learn what the LLM can read.

This is distillation where the teacher has privileged information (the labels)
that the student never sees. The student must recover it from the signal.

STATUS: skeleton. The marked items need answers from the lab before this runs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------- open items
# [ ] What exactly does the teacher see: PTB-XL superclasses, the full SCP
#     statement set, or free-text reports where available?
# [ ] One target summary per RECORD or per 5 s SEGMENT? PTB-XL gives 2 segments
#     per record and they can disagree (observed: atrial flutter 0.62 in one
#     half, absent in the other).
# [ ] Encoder frozen, or fine-tuned end-to-end?
# [ ] Any LoRA on the LLM later, or does it stay fully frozen?
# [ ] Which student LLM? Qwen2.5-0.5B-Instruct is a shape-checking stand-in;
#     ChatHealthAI used Deepseek-R1-Distill-Qwen-14B.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 4
    warmup_steps: int = 100
    max_target_tokens: int = 128
    grad_clip: float = 1.0
    question: str = "Anything wrong with this ECG?"
    device: str = "cuda"


def build_training_example(
    tok,
    llm,
    soft_prompt: torch.Tensor,      # (1, n_queries, d_llm)
    question: str,
    target_text: str,
    device: str = "cuda",
):
    """Assemble inputs_embeds and labels for one example.

    Prompt tokens are masked out of the loss with -100 so that only the
    target summary contributes.
    """
    from .generate import _template_prefix

    emb = llm.get_input_embeddings()

    pre = _template_prefix(tok, "You are given an embedded representation of a 12-lead ECG:")
    suf = "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n"

    ids_pre = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids_suf = tok(suf, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ids_tgt = tok(target_text + "<|im_end|>", return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)

    inputs_embeds = torch.cat(
        [emb(ids_pre), soft_prompt.to(emb.weight.dtype), emb(ids_suf), emb(ids_tgt)],
        dim=1,
    )

    n_prompt = ids_pre.shape[1] + soft_prompt.shape[1] + ids_suf.shape[1]
    labels = torch.full((1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
    labels[0, n_prompt:] = ids_tgt[0]

    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attn, labels


def train(
    aligner: nn.Module,
    encoder,
    tok,
    llm,
    loader: DataLoader,
    targets: Dict[str, str],          # file path -> teacher summary
    cfg: TrainConfig = TrainConfig(),
):
    """Train the aligner. Encoder and LLM stay frozen."""
    for p in encoder.parameters():
        p.requires_grad = False
    for p in llm.parameters():
        p.requires_grad = False
    encoder.eval()
    llm.eval()

    aligner.train()
    aligner.to(cfg.device)
    print(f"trainable parameters: {aligner.n_trainable():,}")

    opt = torch.optim.AdamW(
        aligner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    step = 0
    for epoch in range(cfg.epochs):
        running = []
        for source, inps in loader:
            with torch.no_grad():
                enc = encoder(source=source.to(cfg.device))
                h = enc["encoder_out"]

            soft = aligner(h)                       # (B, n_queries, d_llm)

            losses = []
            for b, inp in enumerate(inps):
                tgt = targets.get(inp.meta.file)
                if tgt is None:
                    continue
                ie, attn, labels = build_training_example(
                    tok, llm, soft[b : b + 1], cfg.question, tgt, cfg.device
                )
                out = llm(inputs_embeds=ie, attention_mask=attn, labels=labels)
                losses.append(out.loss)

            if not losses:
                continue

            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(aligner.parameters(), cfg.grad_clip)
            opt.step()

            running.append(loss.item())
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {sum(running[-20:])/20:.4f}")

        print(f"epoch {epoch} mean loss {sum(running)/max(len(running),1):.4f}")

    return aligner


# ------------------------------------------------------------------ teacher
def build_teacher_prompt(labels: List[str], extra: Optional[str] = None) -> str:
    """Prompt for gpt-oss 120B. It sees the labels; the student never will."""
    label_str = ", ".join(labels)
    prompt = (
        "You are a cardiologist writing a brief ECG interpretation.\n"
        f"Ground-truth findings for this 12-lead ECG: {label_str}\n"
    )
    if extra:
        prompt += f"Additional context: {extra}\n"
    prompt += (
        "\nWrite a two-sentence interpretation for a clinician. State findings "
        "plainly. Do not mention that you were given the findings."
    )
    return prompt
