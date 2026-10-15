"""Contract for the resampler's task-text stream mask.

Two things are pinned here, and they are the reason the mask lives in one shared
helper instead of being written out at each call site:

1. THE COUPLING. ``prompt_only_task_mask`` is ``attention_mask & (labels == -100)``,
   which equals "the real prompt tokens" only because the collate masks the whole
   prompt and nothing but the prompt. That is an agreement between two pieces of
   code, not a property of the tensors. So the prompt length is derived here a
   second time, independently -- as the length of the leading run of -100 -- and the
   two are required to agree. Supervise any part of the prompt and this fails, which
   is the point: the alternative is the task stream silently shortening.

2. TEACHER FORCING IS UNTOUCHED. The stream setting changes what the RESAMPLER
   reads. It must not change what the LLM receives or where the loss lands, so the
   count of loss-carrying positions is asserted identical under every setting.

3. THE QUESTION STREAM IS ITS OWN TEXT. Under ``question`` the resampler reads a
   separate tokenisation of "Question: <paraphrase>\nAnswer:" -- no chat template,
   no patient context -- built the same way by the training and generation
   collates. Under ``none`` it reads nothing. Both are checked against the text
   itself, not against the mask arithmetic.

Tokenisation uses the debug student (Qwen2.5-0.5B), cached locally, so this runs on
CPU. The teacher-forced assertions use a stand-in LLM: a causal running mean plus a
linear head, which is enough for the position bookkeeping being tested and means no
14B has to be present.

    pytest tests/test_task_stream.py
    python  tests/test_task_stream.py
"""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from src.data.dataset import (
    DEFAULT_PREFIX_TEXT,
    TASK_STREAM_MODES,
    DatasetConfig,
    _format_prompt,
    build_datasets,
    load_student_tokenizer,
    make_collate_fn,
    prompt_only_task_mask,
    render_question,
    resampler_question_text,
    task_stream_ids,
)
from src.model.resampler import PerceiverResamplerConfig, Resampler

CONFIG = DatasetConfig(debug=True)
_SPLITS = None


def splits():
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = build_datasets(CONFIG)
    return _SPLITS


def _leading_minus_100_run(labels_row: torch.Tensor) -> int:
    """Prompt length as the leading run of -100 in the labels.

    NOT sufficient on its own, and worth being precise about why. If someone
    supervises the TAIL of the prompt, the run shortens by exactly as much as the
    mask does, and the two agree while the resampler's stream has silently lost
    those positions. The run catches interior supervision; ``_tokenised_prompt_len``
    below is the derivation that catches the rest. Both are asserted.
    """
    run = 0
    for value in labels_row.tolist():
        if value != -100:
            break
        run += 1
    return run


def _tokenised_prompt_len(item, tok, max_length: int) -> int:
    """Prompt length from the tokenizer, touching neither labels nor the mask.

    Rebuilds the prompt string the way :func:`make_collate_fn` builds it and counts
    its tokens. This is the anchor that makes the invariant real: it cannot move
    when the label bookkeeping moves, so a change to what the collate supervises
    shows up as a disagreement rather than as two numbers shifting together.

    ``strip_think`` comes from the config, so this follows the collate wherever that
    default sits; it is not a claim about which setting is right.
    """
    user_content = f"{DEFAULT_PREFIX_TEXT}\n\n{item['task_prompt']}"
    prompt_str = _format_prompt(tok, user_content, CONFIG.strip_think)
    n = len(tok(prompt_str, add_special_tokens=False).input_ids)
    return min(n, max_length)


def _batch(max_length: int, n: int = 4):
    ds = splits()["train"]
    tok = load_student_tokenizer(CONFIG)
    collate = make_collate_fn(tok, max_length=max_length)
    items = [ds[i] for i in range(n)]
    batch = collate(items)
    batch["_prompt_lens"] = [_tokenised_prompt_len(it, tok, max_length) for it in items]
    return batch


def test_task_mask_is_exactly_the_prompt():
    batch = _batch(max_length=1024)
    labels, attn = batch["labels"], batch["attention_mask"]
    mask = prompt_only_task_mask(attn, labels)

    assert mask.dtype == torch.bool
    assert mask.shape == labels.shape
    for r in range(labels.shape[0]):
        tokenised = batch["_prompt_lens"][r]
        run = _leading_minus_100_run(labels[r])
        kept = int(mask[r].sum())
        assert kept == tokenised, (
            f"row {r}: task mask keeps {kept} positions but the prompt tokenises to "
            f"{tokenised}. Either the collate no longer masks exactly the prompt, or "
            "the mask expression has drifted from it -- the resampler's task stream "
            "is not the prompt any more.")
        assert run == tokenised, (
            f"row {r}: the leading -100 run is {run} but the prompt tokenises to "
            f"{tokenised}; the collate is supervising part of the prompt.")
        # ... and it is that prefix, not merely the same number of positions elsewhere.
        assert mask[r][:tokenised].all()
        assert not mask[r][tokenised:].any()


def test_task_mask_excludes_target_and_padding_separately():
    """The two clauses do different jobs; neither is redundant."""
    batch = _batch(max_length=1024)
    labels, attn = batch["labels"], batch["attention_mask"]
    mask = prompt_only_task_mask(attn, labels)

    supervised = labels != -100
    padding = attn == 0
    assert supervised.any() and padding.any(), "need both to test both clauses"
    assert not mask[supervised].any(), "target positions must be excluded"
    assert not mask[padding].any(), "padding must be excluded"
    # attention_mask alone would keep the target; labels alone would keep the padding.
    assert int(attn.bool().sum()) > int(mask.sum())
    assert int((labels == -100).sum()) > int(mask.sum())


def test_task_mask_when_the_target_is_truncated_away_entirely():
    """A prompt longer than max_length leaves a row with no target at all.

    The collate truncates ``(prompt_ids + target_ids)[:max_length]``, so a small
    enough cap supervises nothing: every label is -100 and the whole attended row is
    prompt. The mask must then be the full attended length, and the teacher-forced
    path has to cope with a row carrying zero loss positions.
    """
    batch = _batch(max_length=8, n=4)
    labels, attn = batch["labels"], batch["attention_mask"]
    mask = prompt_only_task_mask(attn, labels)

    assert (labels == -100).all(), (
        "max_length=8 should truncate every target away; if the prompt got shorter "
        "than 8 tokens this test needs a smaller cap")
    for r in range(labels.shape[0]):
        assert (int(mask[r].sum())
                == _leading_minus_100_run(labels[r])
                == int(attn[r].sum())
                == batch["_prompt_lens"][r])
    assert int(mask.sum()) > 0, "a fully truncated row must still expose its prompt"


class _StubStudent(torch.nn.Module):
    """Stands in for the frozen LLM: masked causal running mean, then a linear head.

    Causal rather than position-wise so the spliced prefix can actually influence
    later positions -- with a position-wise head the prefix would be unobservable
    and these tests would pass for the wrong reason.

    It also HONOURS ``attention_mask``, which a toy stand-in is tempted to ignore.
    A model that ignored it would make every attention-mask regression invisible
    here: narrowing the student's own mask to the prompt would change no output and
    no test would fail.
    """

    def __init__(self, vocab: int, embed_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, embed_dim)
        self.head = torch.nn.Linear(embed_dim, vocab)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False):
        x = inputs_embeds.float()
        if attention_mask is None:
            m = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        else:
            m = attention_mask.to(x.dtype)
        weighted = (x * m.unsqueeze(-1)).cumsum(dim=1)
        seen = m.cumsum(dim=1).clamp(min=1.0).unsqueeze(-1)
        out = type("Out", (), {})()
        out.logits = self.head((weighted / seen).to(inputs_embeds.dtype))
        return out


def _stub_pair(batch):
    cfg = PerceiverResamplerConfig.debug()
    torch.manual_seed(0)
    vocab = max(int(batch["input_ids"].max()), int(batch["task_input_ids"].max())) + 1
    model = _StubStudent(vocab, cfg.embed_dim).eval()
    resampler = Resampler(cfg).eval()
    ecg = torch.randn(batch["input_ids"].shape[0], 312, cfg.input_embed_dim)
    ecg = ecg * (350.0 / ecg.norm(dim=-1, keepdim=True))
    return model, resampler, {**batch, "ecg_embed": ecg}


def test_teacher_forcing_loss_positions_survive_every_task_stream():
    """Item 4: the supervised positions are identical under every setting.

    If a future change narrows ``input_ids`` instead of the mask, the target stops
    reaching the LLM and these counts diverge.
    """
    from evaluate import teacher_forced_batch

    batch = _batch(max_length=1024)
    model, resampler, batch = _stub_pair(batch)
    expected = (batch["labels"] != -100).sum(dim=1).tolist()

    per_setting = {}
    for mode in TASK_STREAM_MODES:
        out = teacher_forced_batch(model, resampler, batch, torch.device("cpu"),
                                   task_stream=mode)
        per_setting[mode] = [n for _, n, _ in out]

    assert all(v == per_setting["prompt"] for v in per_setting.values()), (
        f"loss-carrying positions changed with the task stream: {per_setting}")
    assert per_setting["prompt"] == expected, (
        f"teacher forcing supervises {per_setting['prompt']} positions but labels "
        f"carry {expected}; the target is no longer reaching the loss")
    assert all(n > 0 for n in expected)


def test_llm_inputs_are_byte_identical_across_task_streams():
    """The LLM's own tensors must not move with the resampler's stream.

    Counting loss positions is not enough on its own: narrowing the LLM's attention
    mask to the prompt leaves the label count untouched while breaking teacher
    forcing outright. So this asserts the spliced width, the attention mask and the
    token ids directly, and then -- with the prefix zeroed, which removes the only
    legitimate difference between the settings -- requires the losses to come out
    bit-identical. Anything the setting touches downstream of the resampler shows up
    here.
    """
    from evaluate import splice_prefix, teacher_forced_batch

    batch = _batch(max_length=1024)
    model, resampler, batch = _stub_pair(batch)
    dev = torch.device("cpu")
    attn = batch["attention_mask"]
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    assert torch.equal(prompt_only_task_mask(attn, labels),
                       task_stream_ids(batch, "prompt")[1])

    spliced = {}
    for mode in TASK_STREAM_MODES:
        task_ids, task_mask = task_stream_ids(batch, mode)
        _, full_attn, Lq = splice_prefix(model, resampler, batch["ecg_embed"],
                                         input_ids, attn, dev, task_ids=task_ids,
                                         task_mask=task_mask)
        spliced[mode] = (full_attn, Lq)

    a = spliced["prompt"]
    for mode, b in spliced.items():
        assert a[1] == b[1] == resampler.config.resampled_length
        assert torch.equal(a[0], b[0]), (
            f"the LLM's attention mask changed under --task-stream {mode}: the "
            "student's input moved, which is broken teacher forcing, not a "
            "different resampler view")
    assert a[0].shape[1] == Lq + input_ids.shape[1], (
        "the spliced sequence is shorter than prefix + every token: input_ids has "
        "been truncated somewhere")

    zeroed = {}
    for mode in TASK_STREAM_MODES:
        out = teacher_forced_batch(model, resampler, batch, dev, zero_latents=True,
                                   task_stream=mode)
        zeroed[mode] = [loss for loss, _, _ in out]
    assert all(v == zeroed["prompt"] for v in zeroed.values()), (
        "with the prefix zeroed the task stream cannot matter, yet the losses "
        f"differ: {zeroed}. Something other than the resampler's view is moving.")


def test_task_stream_changes_the_latents_but_not_the_supervision():
    """The flag has to actually do something, or it is a no-op dressed as a control."""
    from evaluate import teacher_forced_batch

    batch = _batch(max_length=1024)
    model, resampler, batch = _stub_pair(batch)

    losses = {}
    for mode in TASK_STREAM_MODES:
        out = teacher_forced_batch(model, resampler, batch, torch.device("cpu"),
                                   task_stream=mode)
        losses[mode] = [loss for loss, _, _ in out]

    for a, b in [(a, b) for i, a in enumerate(TASK_STREAM_MODES)
                 for b in TASK_STREAM_MODES[i + 1:]]:
        assert losses[a] != losses[b], (
            f"{a} and {b} gave the same loss: the stream is not reaching the "
            "resampler, so --task-stream is not measuring what it claims to")


def test_the_question_stream_is_the_paraphrase_alone_and_nothing_of_the_patient():
    """What chathealthAI_train.py:1406 feeds its resampler, and not age or sex."""
    ds = splits()["train"]
    tok = load_student_tokenizer(CONFIG)
    items = [ds[i] for i in range(10)]
    batch = make_collate_fn(tok, max_length=1024)(items)
    ids, mask = task_stream_ids(batch, "question")

    assert mask.dtype == torch.bool
    for r, item in enumerate(items):
        text = tok.decode(ids[r][mask[r]], skip_special_tokens=False)
        question = render_question(item["ecg_id"], item["superclass"])
        assert item["question"] == question
        assert text == resampler_question_text(question) == (
            f"Question: {question}\nAnswer:")
        assert "Patient" not in text and "age" not in text and "sex" not in text
        # ... while the LLM still gets the patient line in its own input
        assert "Patient: age" in tok.decode(batch["input_ids"][r])
        # and the padding is the only thing the mask drops
        n = len(tok(text, add_special_tokens=False).input_ids)
        assert int(mask[r].sum()) == n and mask[r][:n].all()


def test_none_is_an_empty_stream_and_reading_nothing_is_reading_nothing():
    """Zero tokens, and the same latents as a stream whose every position is masked."""
    batch = _batch(max_length=1024)
    _, resampler, batch = _stub_pair(batch)
    ids, mask = task_stream_ids(batch, "none")
    B = batch["input_ids"].shape[0]
    assert ids.shape == (B, 0) and mask.shape == (B, 0) and mask.dtype == torch.bool

    d = resampler.config.embed_dim
    with torch.no_grad():
        empty = resampler(batch["ecg_embed"], torch.zeros(B, 0, d),
                          task_mask=mask)
        masked = resampler(batch["ecg_embed"], torch.randn(B, 7, d),
                           task_mask=torch.zeros(B, 7, dtype=torch.bool))
    assert torch.allclose(empty, masked, atol=1e-6)


def test_generation_reads_the_same_question_stream_training_does():
    """The eval collate builds the question stream with the training collate's helper.

    The prompt modes differ between the two by construction (generation has no
    target), but question and none must be the identical tensors, or a question-
    trained resampler is decoded on a stream it never saw.
    """
    from evaluate import make_eval_collate

    ds = splits()["train"]
    tok = load_student_tokenizer(CONFIG)
    items = [ds[i] for i in range(4)]
    train_batch = make_collate_fn(tok, max_length=1024)(items)
    gen_batch = make_eval_collate(tok, max_prompt_length=1024)(items)
    for mode in ("question", "none"):
        for a, b in zip(task_stream_ids(train_batch, mode), task_stream_ids(gen_batch, mode)):
            assert torch.equal(a, b), mode
    # no labels in a generation batch: both prompt modes read the whole prompt there
    for mode in ("prompt", "prompt-and-target"):
        ids, mask = task_stream_ids(gen_batch, mode)
        assert torch.equal(mask, gen_batch["attention_mask"].bool())


def test_an_unknown_mode_is_an_error_not_a_default():
    import pytest
    with pytest.raises(ValueError, match="unknown task stream"):
        task_stream_ids(_batch(max_length=1024), "questions")


def _ckpt_with_meta(tmp_path, meta, name="ckpt_step10.pt"):
    path = str(tmp_path / name)
    payload = {"resampler": {}, "global_step": 10}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)
    return path


def test_a_resume_on_another_task_stream_is_refused(tmp_path):
    import pytest
    import train

    path = _ckpt_with_meta(tmp_path, {"task_stream": "question"})
    train.require_task_stream_match(path, "question")           # same: through
    with pytest.raises(SystemExit) as err:
        train.require_task_stream_match(path, "prompt")
    message = str(err.value)
    assert "task-stream MISMATCH" in message and "recorded in its meta" in message
    assert "trained with --task-stream question" in message
    assert "this run has --task-stream prompt" in message
    assert "Pass --task-stream question" in message


def test_a_checkpoint_without_the_record_resumes_only_as_prompt_and_target(tmp_path):
    """Before the flag the resampler read the whole of input_ids -- not today's default."""
    import pytest
    import train

    path = _ckpt_with_meta(tmp_path, {"strip_think": False})
    train.require_task_stream_match(path, "prompt-and-target")
    with pytest.raises(SystemExit) as err:
        train.require_task_stream_match(path, "prompt")
    assert "NOT recorded" in str(err.value) and "predates the flag" in str(err.value)


def test_evaluation_defaults_to_the_checkpoints_own_stream(tmp_path, capsys):
    from evaluate import resolve_task_stream

    recorded = _ckpt_with_meta(tmp_path, {"task_stream": "question"}, "a.pt")
    assert resolve_task_stream(recorded, None) == "question"
    assert "question (recorded in the checkpoint)" in capsys.readouterr().out
    # an explicit flag still wins, for a deliberate comparison, and says so
    assert resolve_task_stream(recorded, "prompt") == "prompt"
    assert "trained with --task-stream question" in capsys.readouterr().out
    # no record: the default every evaluation used before there was one
    assert resolve_task_stream(_ckpt_with_meta(tmp_path, None, "b.pt"), None) == "prompt"


def test_every_resampler_call_goes_through_task_stream_ids():
    """No script builds a task stream of its own: one place decides, as for the mask."""
    scripts = os.path.join(REPO_ROOT, "scripts")
    for name in ("train.py", "probe_memory.py", "evaluate.py", "overfit_gate.py",
                 "eval_controls.py", "probe_latents.py", "latent_spread.py"):
        source = open(os.path.join(scripts, name), encoding="utf-8").read()
        assert "prompt_only_task_mask(" not in source, name
        assert 'task_stream == "prompt"' not in source, name


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
