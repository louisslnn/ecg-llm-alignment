"""Prompt assembly and generation with a frozen LLM.

Key finding from the baseline experiments: WHERE the soft prompt is inserted
matters enormously.

  - Prepended BEFORE the chat template opening -> generation collapses into
    repeated tokens ("You You You ...").
  - Inserted INSIDE the user turn -> fluent generation.

The chat template starts with <|im_start|>system. Putting arbitrary vectors
before that breaks the model's expectations. Real multimodal implementations
insert signal tokens inside the user turn, which is what `build_inputs` does.
"""

from typing import List, Optional, Tuple

import torch


def load_llm(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    llm.to(device)
    return tok, llm


def _template_prefix(tok, prefix_text: str) -> str:
    """Chat template opening + user prefix, with the turn left OPEN."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prefix_text}],
        add_generation_prompt=False,
        tokenize=False,
    ).rstrip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")]
    return text


def build_inputs(
    tok,
    llm,
    soft_prompt: Optional[torch.Tensor],
    question: str,
    prefix_text: str = "You are given an embedded representation of a 12-lead ECG:",
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assemble inputs_embeds with the soft prompt inside the user turn.

    Pass soft_prompt=None for the text-only control.
    Returns (inputs_embeds, attention_mask).
    """
    emb = llm.get_input_embeddings()

    pre = _template_prefix(tok, prefix_text)
    suf = "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n"

    ids_pre = tok(pre, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
    ids_suf = tok(suf, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)

    parts = [emb(ids_pre)]
    if soft_prompt is not None:
        parts.append(soft_prompt.to(emb.weight.dtype).to(device))
    parts.append(emb(ids_suf))

    inputs_embeds = torch.cat(parts, dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attn


@torch.no_grad()
def generate(
    tok,
    llm,
    soft_prompt: Optional[torch.Tensor],
    question: str = "Anything wrong with this ECG?",
    max_new_tokens: int = 120,
    device: str = "cuda",
) -> str:
    inputs_embeds, attn = build_inputs(tok, llm, soft_prompt, question, device=device)
    out = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    # NOTE: when generating from inputs_embeds, only NEW tokens are returned.
    # When generating from input_ids, the prompt is echoed and must be sliced.
    return tok.decode(out[0], skip_special_tokens=True)


def format_findings(top: List[Tuple[str, float]]) -> str:
    """Serialise ECG-FM labels as text.

    WARNING: including these in the prompt lets the LLM answer WITHOUT using
    the embedding at all, which makes the comparison uninformative. Keep this
    for the "labels leak" ablation only, not for the main pipeline.
    """
    return ", ".join(f"{n} ({p:.2f})" for n, p in top)
