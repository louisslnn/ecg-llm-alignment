#!/usr/bin/env python
"""Train the resampler against the frozen DeepSeek-R1-Distill-Qwen-14B student.

The student is frozen (bf16, sharded across the visible GPUs, use_cache=False); only
the 843M resampler trains (fp32, AdamW). Each step reuses the exact splice validated
in probe_memory.py: resampler latents are prepended to the token embeddings, prompt
and padding are masked to -100, and the loss lands on the teacher target only.

Two more terms come from ChatHealthAI's ICU workflow, and the objective is

    lambda_llm * llm  +  lambda_linear * linear  +  lambda_consistency * consistency

  linear       a one-logit head on the resampler output (src/model/linear_head.py),
               BCE-with-logits against the gold label, positives weighted by n_neg/n_pos
               of that example's superclass in the train split. Its own AdamW group,
               --linear-head-lr. It trains the resampler too: the latents are not
               detached on the way in.
  consistency  the LLM's teacher-forced Yes-vs-No logit at the position before the
               gold Yes/No token, pulled toward sigmoid(linear logit).detach(). The
               target is detached, so this reaches the resampler through the frozen
               LLM and never the head, which is asserted every step.

--lambda-linear and --lambda-consistency are required, so every run states them; 0 and
0 is the LM loss alone, as every run before them. Each term is logged unweighted and
weighted, with the mean |P_llm(Yes) - P_linear(Yes)| over the batch. best.pt is still
chosen on the validation LLM loss.

Checkpoints (resampler + head + optimizer + step/epoch/best) go to --ckpt-dir
(scratch), every --ckpt-every steps and at each epoch end. On startup it resumes
automatically from the latest checkpoint if one exists. Only the last 2 rotating
checkpoints plus best.pt (lowest val LLM loss) are kept; --save-every-epoch also keeps
a weights-only epochNN.pt (~3.4 GB) per epoch.

    # full run with the linear and consistency losses, every epoch kept
    python scripts/train.py --model-dir $MODEL_DIR --ckpt-dir $SCRATCH/ecg/ckpt \
        --lambda-linear 1.0 --lambda-consistency 0.5 --save-every-epoch

    # the gate: overfit 50 examples, then show it three ways on those same 50 --
    # teacher-forced loss, free generation, and the loss at the Yes/No token for
    # each label separately (scripts/overfit_gate.py). Exits 0 only if all three pass.
    # Pass or fail, the resampler the gate judged is left in --ckpt-dir as
    # overfit_final.pt (~3.4 GB), so it can be probed without retraining it.
    python scripts/train.py --model-dir $MODEL_DIR --overfit 50 \
        --lambda-linear 0 --lambda-consistency 0

    # the same gate with the loss on ONE position per example, the Yes/No token
    # after "Conclusion:" (--loss-on decision; the default, all, is the whole
    # target). Prompt, prefix, task stream and tokenisation are identical; only
    # which target positions contribute changes. Every loss this mode logs is the
    # decision-token mean and is labelled so, the real/shuffled/zeroed controls are
    # reported at that token, split by label, and the gate is judged on condition 4
    # alone: the decision token after the prompt and "Conclusion:" with no target
    # prose, which must be low under the real ECG and at least 0.5 nats higher under
    # a donor ECG that answers the other way. The stop rule is condition 4 (a), that
    # token's real loss per label. It needs the stratified set (below): on the head set every record
    # answers every question the same way, and the controls cannot show anything.
    # The free-generation pass is skipped, since condition 2 is not judged here
    # (--no-skip-generation to run it). Give it its own --ckpt-dir: it writes the
    # same file names as any other gate run.
    python scripts/train.py --model-dir $MODEL_DIR --overfit 50 --loss-on decision \
        --overfit-select stratified \
        --lambda-linear 0 --lambda-consistency 0

    # the same, with the Yes/No token trained the way condition 4 asks for it: each
    # target cut to "Conclusion:" and that token, no Evidence and no Reasoning
    # (--loss-on decision-no-context). Same prefix, task stream and everything else;
    # the training loss, the stop rule and condition 4 are then one quantity. Under
    # --loss-on decision the full target sits in context and supplies the answer.
    python scripts/train.py --model-dir $MODEL_DIR --overfit 50 \
        --loss-on decision-no-context --overfit-select stratified \
        --lambda-linear 0 --lambda-consistency 0

    # the same gate on records drawn so that every superclass has at least three
    # Yes and three No among them (--overfit-select stratified; the default, head,
    # is the first N train examples). On the head set the answer can be read off
    # the question alone; see stratified_records for the rule that draws these.
    python scripts/train.py --model-dir $MODEL_DIR --overfit 50 --overfit-select stratified \
        --lambda-linear 0 --lambda-consistency 0

    # full run (auto-resumes), LM loss alone
    python scripts/train.py --model-dir $MODEL_DIR --ckpt-dir $SCRATCH/ecg/ckpt \
        --lambda-linear 0 --lambda-consistency 0

Compute nodes have no internet, so the checkpoint + tokenizer load from --model-dir
with local_files_only=True (pre-downloaded by scripts/setup_narval.sh).
"""

import argparse
import glob
import os
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for probe_memory

from src.data.dataset import (  # noqa: E402
    TASK_STREAM_MODES,
    DatasetConfig,
    build_datasets,
    make_collate_fn,
    task_stream_ids,
)
from src.model.linear_head import AuxiliaryLinearHead  # noqa: E402
from src.model.resampler import (  # noqa: E402
    PerceiverResamplerConfig,
    Resampler,
    mean_embedding_norm,
)
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
from evaluate import (  # noqa: E402
    checkpoint_objective,
    checkpoint_task_stream,
    objective_of,
    require_strip_think_match,
    yes_no_token_ids,
)
from overfit_gate import (  # noqa: E402
    DECISION_MODES,
    GATE_DECISION_LOSS,
    eval_decision_loss,
    eval_mean_loss,
    eval_no_context_loss,
    gate_batches,
    no_context_target,
    run_gate,
)
from probe_memory import (  # noqa: E402  -- reuse the validated splice + checks
    GradFlowError,
    check_gradients,
    gpu_peaks,
    gpu_reset,
    load_student,
    splice_forward,
)
from decision_token import MARKER, decision_index, tokenise_target  # noqa: E402

GiB = 1024 ** 3

LOSS_ON_MODES = ("all", *DECISION_MODES)
OVERFIT_SELECT_MODES = ("head", "stratified")
# --overfit-select stratified: every superclass has at least this many Yes and this
# many No among the chosen records. The gate's definition, changed here, not per run.
STRATIFIED_MIN_PER_ANSWER = 3
DEFAULT_LINEAR_HEAD_LR = 1e-4
# What a preempted run exits with after its emergency checkpoint. Anything but 0:
# Slurm reads 0 as COMPLETED, and a COMPLETED job is never requeued.
PREEMPT_EXIT_CODE = 75                    # EX_TEMPFAIL


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="local student checkpoint dir")
    ap.add_argument("--ckpt-dir", default=os.environ.get("CKPT_DIR", "checkpoints"),
                    help="where to write/resume checkpoints (put this on scratch)")
    # optimisation (values from the reference / ROADMAP)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None,
                    help="overrides the default: 1e-5 for training, 1e-4 in --overfit "
                         "(the gate tests that the pipeline learns, not the real LR)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=2048)  # MAX_TEXT_LEN
    ap.add_argument("--prefix-scale", type=float, default=1.0,
                    help="multiplier on the output LayerNorm's calibration target: "
                         "the gain becomes (scale * mean_embedding_norm) / "
                         "sqrt(embed_dim). 1.0 puts the latents exactly on the "
                         "student's embedding scale; >1 makes the prefix louder. This "
                         "is the swept variable -- see scripts/gate_scale.sh")
    # Architecture ablations. Both LayerNorms are ON by default and are the two
    # departures from the reference resampler; these turn them off one at a time so a
    # change in the gate can be attributed to one of them. See src/model/resampler.py.
    ap.add_argument("--no-input-norm", action="store_true",
                    help="drop the LayerNorm in front of bio_projection "
                         "(input_layer_norm=False), leaving the cached ECG embeddings "
                         "at their raw ~350 scale, as run 1 trained")
    ap.add_argument("--no-output-norm", action="store_true",
                    help="drop the LayerNorm on the resampler output "
                         "(final_output_norm=False), so the latent scale is whatever "
                         "the block stack produces -- the reference behaviour. There "
                         "is then nothing for --prefix-scale to calibrate, so the two "
                         "flags are mutually exclusive unless the scale is 1.0")
    ap.add_argument("--task-stream", choices=TASK_STREAM_MODES, default="prompt",
                    help="what the RESAMPLER's task-text stream may read. 'prompt' "
                         "(default) is the real prompt tokens only, which is what "
                         "generation feeds it. 'prompt-and-target' is what run 1 "
                         "trained on: the target is in the stream, so the resampler "
                         "can read the answer it will not have at inference. "
                         "'question' is a separate tokenisation of the question "
                         "paraphrase alone, \"Question: ...\\nAnswer:\", with no "
                         "patient context, as chathealthAI_train.py feeds its "
                         "resampler. 'none' is no text at all, the zero-text "
                         "ablation. No setting changes what the LLM receives or "
                         "where the loss lands -- only the resampler's second stream. "
                         "Recorded in every checkpoint; a resume must match it")
    ap.add_argument("--loss-on", choices=LOSS_ON_MODES, default="all",
                    help="which target positions the loss is computed on. 'all' "
                         "(default) is every target token. 'decision' is exactly one "
                         "per example: the Yes/No token after \"Conclusion:\". "
                         "Nothing else differs -- same prompt, prefix, task stream, "
                         "tokenisation and -100 masking of prompt and padding. An "
                         "example whose decision token cannot be placed is fatal, "
                         "never skipped. 'decision-no-context' is that same one "
                         "token with the target cut to \"Conclusion:\" and the token "
                         "-- no Evidence, no Reasoning -- which is what gate condition "
                         "4 measures; 'decision' trains it with the whole target in "
                         "context, which supplies the answer. Every loss logged under "
                         "either is the decision-token mean and is labelled so. "
                         "Recorded in every checkpoint; a resume must match it")
    ap.add_argument("--strip-think", action=argparse.BooleanOptionalAction,
                    default=DatasetConfig.strip_think,
                    help="strip the template's trailing open <think> block from the "
                         "prompt. ON by default; --no-strip-think keeps the block, as "
                         "runs 1 and 2 trained. Evaluation must be given the same "
                         "setting or the student is scored on a prompt it never saw")
    # the linear and consistency losses (ChatHealthAI's ICU workflow)
    ap.add_argument("--lambda-llm", type=float, default=1.0,
                    help="weight on the LM loss, the --loss-on quantity (default 1.0)")
    ap.add_argument("--lambda-linear", type=float, required=True,
                    help="weight on the linear head's BCE against the gold label, "
                         "positives weighted per superclass by n_neg/n_pos in the train "
                         "split. REQUIRED, so a run always states it; 0 builds no head")
    ap.add_argument("--lambda-consistency", type=float, required=True,
                    help="weight on the BCE pulling the LLM's teacher-forced Yes-vs-No "
                         "logit, at the position before the gold Yes/No token, toward "
                         "sigmoid(linear logit).detach(). REQUIRED; > 0 needs "
                         "--lambda-linear > 0, since the target comes from the head")
    ap.add_argument("--linear-head-lr", type=float, default=None,
                    help=f"the head's own AdamW learning rate (default "
                         f"{DEFAULT_LINEAR_HEAD_LR:g}; the resampler's is --lr)")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="also write a weights-only epochNN.pt (resampler + head, no "
                         "optimizer, ~3.4 GB) at each epoch end, never rotated, so the "
                         "run can be evaluated at every epoch and not only the last two")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpointing", choices=["on", "off"], default="off",
                    help="LLM activation checkpointing; off by default -- the probe "
                         "showed batch 4 fits (28.5/21.0 GiB) and measured it a no-op")
    # cadence
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--prefix-log-every", type=int, default=50,
                    help="in --overfit mode, log the latent/token norm ratio and "
                         "||gain|| every N steps, to see the trainable gain drift")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    # the gate
    ap.add_argument("--overfit", type=int, default=None,
                    help="overfit N train examples (no val); the pipeline gate. Which N "
                         "is --overfit-select")
    ap.add_argument("--overfit-select", choices=OVERFIT_SELECT_MODES, default="head",
                    help="which examples --overfit N takes. 'head' (default) is the "
                         "first N of the train split, as every gate run before this "
                         "flag. 'stratified' is N/5 whole records drawn, by a seeded "
                         "rule (--seed), so that each superclass has at least "
                         f"{STRATIFIED_MIN_PER_ANSWER} Yes and {STRATIFIED_MIN_PER_ANSWER} "
                         "No among them: on a set where every record has the same "
                         "five answers, the decision is a function of the question and "
                         "the gate can pass without the ECG being read")
    ap.add_argument("--overfit-target-loss", type=float, default=0.05,
                    help="stop the overfit loop once the eval-mode mean loss over ALL "
                         "the overfit examples reaches this -- gate condition 1's own "
                         "quantity, so it must sit below --gate-threshold. It only "
                         "decides when training stops; the verdict is the "
                         "three-condition gate that runs afterwards. Under --loss-on "
                         "decision it is the eval-mode NO-CONTEXT decision-token "
                         "loss (prompt, then \"Conclusion:\" alone) that must reach "
                         "this, on label=True and label=False separately -- condition "
                         "4 (a)'s quantity, whose threshold is fixed at "
                         f"{GATE_DECISION_LOSS:g}")
    ap.add_argument("--overfit-eval-every", type=int, default=50,
                    help="check that stop rule every N steps (one no-grad pass over "
                         "the overfit examples each time)")
    ap.add_argument("--overfit-max-steps", type=int, default=3000)
    ap.add_argument("--gate-threshold", type=float, default=0.1,
                    help="in --overfit mode, the mean teacher-forced loss over the "
                         "overfit examples must be below this. One of three gate "
                         "conditions; see scripts/overfit_gate.py for the other two. "
                         "Under --loss-on decision or decision-no-context this "
                         "condition is printed against the threshold but not judged")
    ap.add_argument("--skip-generation", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="skip the gate's free-generation pass, condition 2. ON by "
                         "default under --loss-on decision and decision-no-context, "
                         "where nothing supervises "
                         "the prose, condition 2 is not judged, and the pass is most "
                         "of the gate's time; --no-skip-generation runs it anyway. "
                         "OFF otherwise, and refused under --loss-on all, where "
                         "condition 2 is judged and the gate could not pass without it")
    # data path overrides (default to DatasetConfig)
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    args = ap.parse_args()
    # With no output LayerNorm there is no calibration target and no gain to set, so a
    # non-unit scale would be silently ignored -- and the run would carry a label
    # (log name, checkpoint dir, checkpoint meta) claiming a scale it never applied.
    if args.no_output_norm and args.prefix_scale != 1.0:
        ap.error("--prefix-scale has nothing to calibrate with --no-output-norm; "
                 f"drop one of the two (got --prefix-scale {args.prefix_scale:g})")
    # The loop stops on the number condition 1 then re-measures. A stop target at or
    # above the gate's threshold would stop a run the gate is bound to fail.
    # Under either decision mode the loop stops on condition 4 (a)'s number
    # instead, and the same holds against that condition's threshold.
    if args.overfit is not None and args.loss_on in DECISION_MODES:
        if args.overfit_target_loss >= GATE_DECISION_LOSS:
            ap.error(f"--overfit-target-loss ({args.overfit_target_loss:g}) must be "
                     f"below the gate's decision-token threshold "
                     f"({GATE_DECISION_LOSS:g}): under --loss-on {args.loss_on} the loop "
                     "stops on the same quantity the gate's condition 4 (a) checks")
    elif args.overfit is not None and args.overfit_target_loss >= args.gate_threshold:
        ap.error(f"--overfit-target-loss ({args.overfit_target_loss:g}) must be below "
                 f"--gate-threshold ({args.gate_threshold:g}): the loop stops on the "
                 "same quantity the gate's first condition checks")
    if args.overfit_eval_every < 1:
        ap.error("--overfit-eval-every must be at least 1")
    lambdas = {"--lambda-llm": args.lambda_llm, "--lambda-linear": args.lambda_linear,
               "--lambda-consistency": args.lambda_consistency}
    negative = [f"{k} {v:g}" for k, v in lambdas.items() if v < 0]
    if negative:
        ap.error(f"loss weights cannot be negative: {', '.join(negative)}")
    if not any(lambdas.values()):
        ap.error("all three loss weights are 0: nothing would train")
    # The consistency target is sigmoid of the head's logit. With no linear loss the
    # head never trains, and the LLM would be pulled toward its initialisation.
    if args.lambda_consistency > 0 and args.lambda_linear == 0:
        ap.error(f"--lambda-consistency {args.lambda_consistency:g} needs "
                 "--lambda-linear > 0: the consistency target comes from the linear "
                 "head, which nothing would train")
    if args.linear_head_lr is not None and args.lambda_linear == 0:
        ap.error("--linear-head-lr sets the linear head's learning rate; with "
                 "--lambda-linear 0 there is no head and it would be silently ignored")
    if args.linear_head_lr is None:
        args.linear_head_lr = DEFAULT_LINEAR_HEAD_LR
    if args.overfit is not None:
        # The gate's stop rule and verdict are about the LM loss alone.
        if args.lambda_linear or args.lambda_consistency or args.lambda_llm != 1.0:
            ap.error("--overfit trains and judges the LM loss alone: pass "
                     "--lambda-llm 1 --lambda-linear 0 --lambda-consistency 0 (got "
                     + ", ".join(f"{k} {v:g}" for k, v in lambdas.items()) + ")")
        if args.save_every_epoch:
            ap.error("--save-every-epoch writes a checkpoint per epoch of a full run; "
                     "--overfit has no epochs to keep and would silently ignore it")
    # On the head set all ten records are normal: every question has one answer
    # across the set, so condition 4's shuffled column and donor rows cannot show
    # anything, and its verdict would be meaningless rather than merely weak.
    if (args.overfit is not None and args.loss_on in DECISION_MODES
            and args.overfit_select == "head"):
        ap.error(f"--loss-on {args.loss_on} cannot run with --overfit-select head (the "
                 "default): all ten head records answer each question the same way, "
                 "so condition 4's shuffled column and donor-answers-opposite rows "
                 "cannot show anything and its verdict would be meaningless. Pass "
                 "--overfit-select stratified")
    if args.skip_generation is None:
        args.skip_generation = (args.overfit is not None
                                and args.loss_on in DECISION_MODES)
    elif args.skip_generation and args.overfit is None:
        ap.error("--skip-generation skips a pass of the --overfit gate; without "
                 "--overfit N it would be silently ignored")
    elif args.skip_generation and args.loss_on == "all":
        ap.error("--skip-generation is refused under --loss-on all: condition 2, free "
                 "generation, is judged there, and the gate cannot pass without it")
    if args.overfit_select == "stratified":
        # The unit is a record with its five questions, and three Yes plus three No
        # for one superclass already takes six records.
        min_n = 2 * STRATIFIED_MIN_PER_ANSWER * len(SUPERCLASSES)
        if args.overfit is None:
            ap.error("--overfit-select stratified selects the --overfit examples; "
                     "without --overfit N it would be silently ignored")
        if args.overfit % len(SUPERCLASSES) or args.overfit < min_n:
            ap.error(f"--overfit-select stratified needs --overfit to be a whole "
                     f"number of records times {len(SUPERCLASSES)} questions, and at "
                     f"least {min_n} (got --overfit {args.overfit})")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Set by the SIGUSR1/SIGTERM handler (train.sh's `--signal=USR1@120` fires ~120 s
# before the time limit / on preemption). The training loop checkpoints at the next
# step boundary and calls requeue_and_exit, which restarts it from that checkpoint.
_PREEMPT = {"flag": False}

# Settings written into every checkpoint, filled in by main(). A prefix-scale sweep
# yields checkpoints whose weights differ only through one scalar; without this there
# is nothing in the file that says which run produced it.
_RUN_META: dict = {}


def git_state():
    """``(head, dirty, lines)``: the commit this code is at and how the tree differs.

    ``dirty`` means tracked files differ from HEAD. Untracked paths are listed but do
    not make the tree dirty on their own, or a logs/ directory would mark every run;
    read the list, since an untracked module this script imports is still code that
    no commit records. ``head`` is None when git cannot answer.
    """
    def git(*cmd):
        return subprocess.run(["git", "-C", REPO_ROOT, *cmd], capture_output=True,
                              text=True, check=True).stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        lines = git("status", "--porcelain").splitlines()
    except (OSError, subprocess.CalledProcessError) as err:
        return None, None, [f"git unavailable: {err}"]
    return head, any(not line.startswith("??") for line in lines), lines


def log_git_state():
    head, dirty, lines = git_state()
    if head is None:
        print(f"git HEAD unknown ({lines[0]})")
    else:
        n_untracked = sum(line.startswith("??") for line in lines)
        state = (f"DIRTY: {len(lines) - n_untracked} tracked file(s) differ from HEAD"
                 if dirty else "clean")
        if n_untracked:
            state += f", {n_untracked} untracked path(s)"
        print(f"git HEAD {head}  working tree {state}")
        for line in lines:
            print(f"    {line}")
    sys.stdout.flush()
    _RUN_META.update({"git_head": head, "git_dirty": dirty})


def _install_signal_handlers():
    def handler(signum, _frame):
        _PREEMPT["flag"] = True
    signal.signal(signal.SIGUSR1, handler)
    signal.signal(signal.SIGTERM, handler)


def requeue_and_exit():
    """After the emergency checkpoint: put the job back in the queue, exit non-zero.

    Exiting 0 here is what lost a run: Slurm records exit 0 as COMPLETED, --requeue
    never fires for a COMPLETED job, and the time-limit signal is not a preemption in
    any case. So the job requeues itself with ``scontrol requeue``, and exits
    PREEMPT_EXIT_CODE whatever happened, so a requeue that failed shows as FAILED in
    sacct rather than as a run that finished. Outside Slurm there is nothing to
    requeue; the exit code alone says the run stopped early.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        try:
            r = subprocess.run(["scontrol", "requeue", job], capture_output=True,
                               text=True, timeout=60)
            ok, said = r.returncode == 0, (r.stderr or r.stdout).strip()
        except (OSError, subprocess.SubprocessError) as err:
            ok, said = False, str(err)
        print(f"scontrol requeue {job}: " + ("requeued" if ok else
              f"FAILED ({said}); resubmit by hand, the checkpoint is written"),
              flush=True)
    else:
        print("no SLURM_JOB_ID: not under Slurm, nothing to requeue", flush=True)
    sys.exit(PREEMPT_EXIT_CODE)


def preflight(args, data_cfg):
    """Check every input path exists BEFORE loading the 14B model (slow)."""
    checks = [
        ("model dir", args.model_dir),
        ("model config", os.path.join(args.model_dir, "config.json")),
        ("embedding index", os.path.join(data_cfg.emb_cache_dir, "index.json")),
        ("manifest", data_cfg.manifest_path),
        ("teacher jsonl", data_cfg.teacher_path),
    ]
    missing = [f"{name}: {path}" for name, path in checks if not os.path.exists(path)]
    if missing:
        sys.exit("preflight failed, missing input(s):\n  " + "\n  ".join(missing))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if not os.access(args.ckpt_dir, os.W_OK):
        sys.exit(f"preflight failed: ckpt dir not writable: {args.ckpt_dir}")
    print("preflight OK; all inputs present, ckpt dir writable")


# --------------------------------------------------------------------------- #
# Checkpointing                                                               #
# --------------------------------------------------------------------------- #

_STEP_RE = re.compile(r"ckpt_step(\d+)\.pt$")
OVERFIT_FINAL = "overfit_final.pt"


def _rotating(ckpt_dir):
    return sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt")),
                  key=lambda p: int(_STEP_RE.search(p).group(1)))


def _payload(resampler, epoch, step, best_val, meta=None, head=None):
    """What every checkpoint carries; evaluate.py reads these keys. ``linear_head``
    is there exactly when the run has a head (its meta's lambda_linear > 0), which
    evaluate.load_linear_head holds a checkpoint to."""
    payload = {
        "resampler": resampler.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_val_loss": best_val,
        # what this run was: a sweep over prefix_scale produces checkpoints that are
        # otherwise indistinguishable, so each one carries its own settings
        "meta": dict(_RUN_META) if meta is None else meta,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if head is not None:
        payload["linear_head"] = head.state_dict()
    return payload


def save_checkpoint(ckpt_dir, resampler, optimizer, epoch, step, best_val, is_best,
                    meta=None, head=None):
    payload = _payload(resampler, epoch, step, best_val, meta, head)
    payload["optimizer"] = optimizer.state_dict()
    path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic
    if is_best:
        best = os.path.join(ckpt_dir, "best.pt")
        torch.save(payload, best + ".tmp")
        os.replace(best + ".tmp", best)
    # keep only the last 2 rotating checkpoints (best.pt is separate, never pruned)
    for old in _rotating(ckpt_dir)[:-2]:
        os.remove(old)
    return path


def save_epoch_checkpoint(ckpt_dir, resampler, epoch, step, best_val, head=None):
    """--save-every-epoch: ``ckpt_dir/epochNN.pt`` after ``epoch`` epochs, never pruned.

    Weights and meta only, no optimizer state, ~3.4 GB: enough to evaluate the run at
    that epoch, not to resume it (resume() only looks at ckpt_step*.pt). The rotation
    keeps two checkpoints, so without these the epoch at which the head and the LLM
    agree best cannot be found after the fact.
    """
    payload = _payload(resampler, epoch, step, best_val, head=head)
    path = os.path.join(ckpt_dir, f"epoch{epoch:02d}.pt")
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic
    return path


def save_overfit_final(ckpt_dir, resampler, epoch, step, gate_passed, args):
    """Write the resampler the gate just judged to ``ckpt_dir/overfit_final.pt``.

    Written on a failed gate as much as a passed one: the failing resampler is the
    one worth probing, and without this it has to be retrained to ask it anything
    the gate did not. The payload is a training checkpoint's -- the same keys, the
    same meta with strip_think -- so evaluate.load_resampler and
    evaluate.checkpoint_strip_think read it as they read any other, plus the gate's
    verdict, the run's full arguments and the examples the verdict is about
    (``overfit_selection``, the meta's entry from select_overfit: mode, ecg_ids,
    Yes/No counts per superclass). It has no optimizer state: nothing resumes
    from it (resume() only looks at ckpt_step*.pt), and AdamW's two moments would
    triple the ~3.4 GB the fp32 weights take. best_val_loss is inf, as in any
    checkpoint written before a validation pass: --overfit has no validation.
    """
    payload = _payload(resampler, epoch, step, float("inf"))
    payload["gate_passed"] = bool(gate_passed)
    payload["args"] = dict(vars(args))
    payload["overfit_selection"] = payload["meta"].get("overfit_selection")
    path = os.path.join(ckpt_dir, OVERFIT_FINAL)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic
    return path


def require_task_stream_match(path, task_stream):
    """Refuse to resume a checkpoint on a resampler stream it did not train on.

    Read from the meta with mmap, before the student loads. A checkpoint with no
    entry predates --task-stream (124b2a2), when the resampler read the whole of
    input_ids: that is prompt-and-target, NOT today's default. There is no way
    through: the resampler's second input changing mid-run is not an experiment.
    """
    recorded = checkpoint_task_stream(path)
    trained = recorded or "prompt-and-target"
    if trained == task_stream:
        return
    source = ("recorded in its meta" if recorded else
              "NOT recorded: the checkpoint predates the flag, when the resampler "
              "read the whole of input_ids")
    sys.exit(f"task-stream MISMATCH: {path} was trained with --task-stream {trained} "
             f"({source}), this run has --task-stream {task_stream}. Resuming would "
             "hand the resampler a second input it never trained on. Pass "
             f"--task-stream {trained}, or a fresh --ckpt-dir.")


def objective_lambdas(args):
    """This run's three loss weights, keyed as the checkpoint meta records them."""
    return {"lambda_llm": float(args.lambda_llm),
            "lambda_linear": float(args.lambda_linear),
            "lambda_consistency": float(args.lambda_consistency)}


def require_objective_match(path, trained, recorded, lambdas):
    """Refuse to resume a checkpoint under loss weights it did not train with.

    ``(trained, recorded)`` is evaluate.objective_of on the checkpoint's meta, so a
    checkpoint from before the flags is the LM loss alone (1, 0, 0). A change of
    weights is a different objective -- and a different optimizer, when the head
    comes or goes -- so there is no way through, as for --loss-on.
    """
    if trained == lambdas:
        return
    flags = "  ".join(f"--{k.replace('_', '-')} {v:g}" for k, v in trained.items())
    sys.exit(f"objective MISMATCH: {path} was trained with {flags} ("
             + ("recorded in its meta" if recorded else "NOT recorded: it predates the "
                "flags, so the LM loss alone") + "), this run has "
             + "  ".join(f"--{k.replace('_', '-')} {v:g}" for k, v in lambdas.items())
             + f". Pass {flags}, or a fresh --ckpt-dir.")


def resume(ckpt_dir, resampler, optimizer, dev, loss_on="all", head=None,
           lambdas=None):
    ckpts = _rotating(ckpt_dir)
    if not ckpts:
        print("no checkpoint found; starting fresh")
        return 0, 0, float("inf")
    path = ckpts[-1]
    # our own checkpoint (trusted); it carries numpy/python RNG state that the
    # weights_only=True default (torch>=2.6) refuses to unpickle.
    payload = torch.load(path, map_location=dev, weights_only=False)
    # A checkpoint with no entry predates --loss-on and trained on the whole target.
    trained_on = (payload.get("meta") or {}).get("loss_on", "all")
    if trained_on != loss_on:
        sys.exit(f"loss-on MISMATCH: {path} was trained with --loss-on {trained_on}, "
                 f"this run has --loss-on {loss_on}. Resuming would continue one "
                 "objective under another, against a best_val_loss that is a "
                 f"different quantity. Pass --loss-on {trained_on}, or a fresh "
                 "--ckpt-dir.")
    if lambdas is not None:
        require_objective_match(path, *objective_of(payload.get("meta")), lambdas)
    resampler.load_state_dict(payload["resampler"])
    if head is not None:
        head.load_state_dict(payload["linear_head"])
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    print(f"resumed from {os.path.basename(path)}: epoch {payload['epoch']} "
          f"step {payload['global_step']} best_val {payload['best_val_loss']:.4f}")
    return payload["epoch"], payload["global_step"], payload["best_val_loss"]


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

class DecisionTokenError(RuntimeError):
    """--loss-on decision could not place its one supervised position for an example."""


def make_decision_collate(tokenizer, collate):
    """``collate``, plus ``decision_mask``: the one position --loss-on decision trains on.

    Wraps training's collate and leaves every tensor it produced as it was. In
    particular ``labels`` still carries the whole target: prompt_only_task_mask reads
    the prompt off ``labels == -100``, so narrowing labels here would move the
    target into the resampler's task stream. The narrowing happens in
    probe_memory.splice_forward, after that mask is taken; this only says where.

    ``decision_mask`` is bool, shaped like ``labels``, with exactly one True per row:
    the Yes/No token after "Conclusion:", found with decision_token's locator on the
    target rebuilt by tokenise_target -- the pair the gate's condition 3 uses, and
    the same guard. The rebuilt target must be the supervised span token for token
    (so a target --max-length cut short fails here, where --loss-on all would train
    on what was left), and the decision token must be locatable. Either failure
    raises DecisionTokenError naming the examples. Nothing is skipped: with one
    position per example, dropping an example changes what the loss is a mean of.
    """
    def decision_collate(batch):
        out = collate(batch)
        labels = out["labels"]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        problems = []
        for i, b in enumerate(batch):
            span = (labels[i] != -100).nonzero().flatten()   # the supervised positions
            ids, offsets = tokenise_target(tokenizer, b["target"])
            k, why = decision_index(b["target"], offsets)
            if len(ids) != len(span):
                why = (f"rebuilt target has {len(ids)} tokens, the supervised span "
                       f"{len(span)}"
                       + (" (cut short by --max-length)" if len(span) < len(ids) else ""))
            elif labels[i][span].tolist() != ids:
                why = "rebuilt target has the supervised span's length but not its tokens"
            if why:
                problems.append(f"  {b['ecg_id']:>7} {b['superclass']:<6} {why}")
                continue
            mask[i, span[k]] = True
        if problems:
            raise DecisionTokenError(
                f"the token after {MARKER!r} cannot be placed for {len(problems)} of "
                f"the {len(batch)} examples in a batch:\n" + "\n".join(problems)
                + "\nNothing was skipped: the loss is one position per example, so an "
                "example left out makes it a mean over different examples.")
        out["decision_mask"] = mask
        return out

    return decision_collate


def make_no_context_collate(tokenizer, collate):
    """--loss-on decision-no-context's collate: each target cut to "Conclusion:" and
    its Yes/No token, then :func:`make_decision_collate` over the cut.

    The cut is overfit_gate.no_context_target's, from the target's own text. So the
    prompt, the template, strip_think, the question stream and the -100 masking are
    training's as ever, the batch is what make_decision_collate makes of it with
    every one of its checks, and the one supervised position sees "Conclusion:" and
    nothing of the Evidence or Reasoning before it -- exactly what gate condition 4
    shows the model. The marked token must also be the one condition 4 scores, id
    for id. Any example that fails either raises DecisionTokenError naming it;
    nothing is skipped, for the reason make_decision_collate gives.
    """
    decision_collate = make_decision_collate(tokenizer, collate)

    def no_context_collate(batch):
        cut, want, problems = [], [], []
        for b in batch:
            target, decision_id, why = no_context_target(tokenizer, b["target"])
            if why:
                problems.append(f"  {b['ecg_id']:>7} {b['superclass']:<6} {why}")
                continue
            cut.append({**b, "target": target})
            want.append(decision_id)
        if problems:
            raise DecisionTokenError(
                f"the target cannot be cut to {MARKER!r} and its decision token for "
                f"{len(problems)} of the {len(batch)} examples in a batch:\n"
                + "\n".join(problems))
        out = decision_collate(cut)
        got = out["labels"][out["decision_mask"]].tolist()   # one per row, in order
        if got != want:
            raise DecisionTokenError(
                f"the marked tokens {got} are not the decision tokens condition 4 "
                f"scores, {want}")
        return out

    return no_context_collate


class ConsistencyTargetError(RuntimeError):
    """A target the consistency loss cannot read its gold Yes/No position from."""


_SWAP = {"yes": "no", "no": "yes"}


def _swapped_word(word):
    """The other answer, spelled in ``word``'s case: Yes -> No, no -> yes, YES -> NO."""
    other = _SWAP[word.lower()]
    if word.isupper() and len(word) > 1:
        return other.upper()
    return other.capitalize() if word[0].isupper() else other


def consistency_position(tokenizer, target, label, yes_ids, no_ids):
    """``(k, ids, why)``: the gold Yes/No token's index in the target's tokens.

    The consistency loss reads the LLM's logits at the row before it. ``ids`` are the
    target's tokens as tokenise_target builds them (the collate's, EOS included).
    The position comes from decision_token's locator -- the one every other script
    uses -- and the reference's guards, every one fatal for the caller:

      * exactly one "Conclusion:" in the target (decision_index);
      * the token after it is a single Yes/No token: one of yes_ids / no_ids
        (evaluate.yes_no_token_ids, the readout's own sets), nothing glued to it;
      * that word agrees with the gold label;
      * the counterfactual: the same target with Yes and No swapped, re-tokenised,
        has its answer at the same index, behind the same tokens, as the other
        word's token. So where the logit is read cannot depend on the answer.

    ``why`` names the first guard that fails, None when all hold. Pure on the
    tokenizer and the text; the collate checks the layout on top of this.
    """
    ids, offsets = tokenise_target(tokenizer, target)
    k, why = decision_index(target, offsets)
    if why:
        return None, ids, why
    word = tokenizer.decode([ids[k]]).strip()
    if word.lower() not in _SWAP or ids[k] not in (*yes_ids, *no_ids):
        return None, ids, (f"the token after {MARKER!r} is "
                           f"{tokenizer.decode([ids[k]])!r}, not a single Yes/No token")
    if (ids[k] in yes_ids) != bool(label):
        return None, ids, (f"the target concludes {word!r} but the gold label is "
                           f"{bool(label)}")
    s, e = offsets[k]
    other = _swapped_word(word)
    swapped = target[:s] + target[s:e].replace(word, other, 1) + target[e:]
    ids2, offsets2 = tokenise_target(tokenizer, swapped)
    k2, why2 = decision_index(swapped, offsets2)
    if (why2 or k2 != k or ids2[:k] != ids[:k]
            or ids2[k] not in (no_ids if label else yes_ids)):
        return None, ids, (f"swapping {word!r} for {other!r} moves the answer: token "
                           f"{k} becomes " + (why2 if why2 else
                           f"{k2}, {tokenizer.decode([ids2[k2]])!r}"))
    return k, ids, None


def make_consistency_collate(tokenizer, collate, yes_ids, no_ids):
    """``collate``, plus ``consistency_pos``: where the consistency loss reads the LLM.

    One long per row: the index in ``input_ids`` of the gold Yes/No token, from
    :func:`consistency_position` on the row's own target, checked against what the
    collate laid out -- the target's tokens must be the supervised span's, token
    for token as far as the span runs, so the gold token is supervised (its label
    is the token, not -100). The logits row before it is the one the loss reads.

    -1 where --max-length cut the target before the gold token: that row has no
    conclusion to read, so it sits out the consistency loss and keeps the LLM and
    linear ones, and the training loop counts it. Any other failure raises
    ConsistencyTargetError naming every failing example in the batch. Every tensor
    ``collate`` built is left as it was.

    Under --loss-on decision or decision-no-context this goes INSIDE their collate,
    so it sees the targets they hand the base collate (cut, for no-context).
    """
    yes_ids, no_ids = list(yes_ids), list(no_ids)

    def consistency_collate(batch):
        out = collate(batch)
        labels = out["labels"]
        pos = torch.full((len(batch),), -1, dtype=torch.long)
        problems = []
        for i, b in enumerate(batch):
            k, ids, why = consistency_position(tokenizer, b["target"], b["label"],
                                               yes_ids, no_ids)
            if why is None:
                span = (labels[i] != -100).nonzero().flatten()
                if labels[i][span].tolist() != ids[:len(span)]:
                    why = (f"the supervised span ({len(span)} tokens) is not the "
                           f"target's tokens ({len(ids)})")
                elif k < len(span):
                    pos[i] = span[k]
            if why:
                problems.append(f"  {b['ecg_id']:>7} {b['superclass']:<6} {why}")
        if problems:
            raise ConsistencyTargetError(
                f"the consistency loss cannot read the gold Yes/No position for "
                f"{len(problems)} of the {len(batch)} examples in a batch:\n"
                + "\n".join(problems))
        out["consistency_pos"] = pos
        return out

    return consistency_collate


def check_consistency_targets(tokenizer, splits, yes_ids, no_ids):
    """Every target of ``splits`` (name -> dataset) through consistency_position, now.

    The collate would raise on the same targets, but hours into an epoch; this says
    all of them at once, before the student loads. Truncation by --max-length is
    not checked here: that depends on the prompt, is not an error, and is counted
    as training meets it.
    """
    t0 = time.time()
    problems, n = [], 0
    for name, dataset in splits.items():
        for e in dataset.examples:
            n += 1
            _, _, why = consistency_position(tokenizer, e.target, e.label,
                                             yes_ids, no_ids)
            if why:
                problems.append(f"  {name:<5} {e.ecg_id:>7} {e.superclass:<6} {why}")
    if problems:
        raise ConsistencyTargetError(
            f"{len(problems)} of {n} targets fail the consistency loss's guards:\n"
            + "\n".join(problems[:50])
            + (f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""))
    print(f"consistency guards: all {n} targets of {', '.join(splits)} pass "
          f"({time.time() - t0:.0f}s)", flush=True)


def superclass_pos_weights(examples):
    """superclass -> n_neg / n_pos over ``examples``: the linear loss's pos_weight.

    One weight per superclass rather than the reference's one global n_neg/n_pos:
    the positive rate runs from 0.415 for NORM to 0.119 for HYP, and the head is
    trained on all five questions at once. Raises ValueError for a superclass with
    no positive, whose weight would be infinite.
    """
    out = {}
    for sc, n in label_counts(examples).items():
        if n["yes"] == 0:
            raise ValueError(f"{sc} has no positive example ({n['no']} negatives): "
                             "its pos_weight n_neg/n_pos is undefined")
        out[sc] = n["no"] / n["yes"]
    return out


def print_pos_weights(examples, weights):
    counts = label_counts(examples)
    print(f"linear loss pos_weight per superclass (train split): "
          f"{'superclass':<10} {'Yes':>6} {'No':>6} {'rate':>6} {'pos_weight':>10}")
    for sc, w in weights.items():
        n = counts[sc]
        print(f"{'':<54}{sc:<10} {n['yes']:>6} {n['no']:>6} "
              f"{n['yes'] / (n['yes'] + n['no']):>6.3f} {w:>10.3f}")
    sys.stdout.flush()


def label_counts(examples):
    """superclass -> {"yes": n, "no": n} over ``examples``, in SUPERCLASSES order."""
    counts = {sc: {"yes": 0, "no": 0} for sc in SUPERCLASSES}
    for e in examples:
        counts[e.superclass]["yes" if e.label else "no"] += 1
    return counts


def stratified_records(examples, n_records, seed):
    """--overfit-select stratified: ``n_records`` ecg_ids, in the order drawn.

    Every superclass ends with at least STRATIFIED_MIN_PER_ANSWER Yes and as many No
    among the chosen records. THE RULE, which is all there is to it:

      1. The candidates are the records in ``examples`` that have all five questions
         (a record the dataset dropped a block from is not a whole unit), in
         ascending ecg_id.
      2. They are shuffled once, by ``random.Random(seed)`` -- its own generator, so
         the streams training draws from are where --overfit-select head leaves them.
      3. ``n_records`` times: a record's score is how many of its five answers are
         one some superclass is still short of. Take the highest score; among equal
         scores, the record earliest in the shuffled order.
      4. If any superclass is still short when the records run out, raise. Nothing is
         relaxed: a set that does not meet the floor is not this selection.

    Nothing is searched and nothing is scored against the model, so the same examples
    and seed give the same records. Once no answer is short every score is zero and
    step 3 is simply the next record in the shuffled order.

    What it favours, on purpose: at ten records the floor asks for fifteen Yes, so
    step 3 reaches for records positive for several superclasses at once. The set is
    balanced, NOT representative -- the Yes/No table printed at startup is what it
    is, and the train split's rates are not.

    Pure on the examples; raises ValueError when the floor cannot be met.
    """
    answers = {}
    for e in examples:
        answers.setdefault(int(e.ecg_id), {})[e.superclass] = bool(e.label)
    order = sorted(r for r, a in answers.items() if len(a) == len(SUPERCLASSES))
    if len(order) < n_records:
        raise ValueError(f"{n_records} records asked for, {len(order)} have all "
                         f"{len(SUPERCLASSES)} questions")
    random.Random(seed).shuffle(order)

    short = {(sc, answer): STRATIFIED_MIN_PER_ANSWER
             for sc in SUPERCLASSES for answer in (True, False)}
    chosen = []
    while len(chosen) < n_records:
        if not any(n > 0 for n in short.values()):
            chosen += order[:n_records - len(chosen)]
            break
        # max() keeps the first of equal scores: the earliest in the shuffled order
        best = max(order, key=lambda r: sum(short[cell] > 0
                                            for cell in answers[r].items()))
        order.remove(best)
        chosen.append(best)
        for cell in answers[best].items():
            short[cell] -= 1
    missing = [f"{sc} {'Yes' if answer else 'No'} has {STRATIFIED_MIN_PER_ANSWER - n}"
               for (sc, answer), n in short.items() if n > 0]
    if missing:
        raise ValueError(
            f"{n_records} records did not give every superclass "
            f"{STRATIFIED_MIN_PER_ANSWER} Yes and {STRATIFIED_MIN_PER_ANSWER} No under "
            f"this rule with seed {seed}: {', '.join(missing)}. Raise --overfit; the "
            "rule is not relaxed to fit")
    return chosen


def select_overfit(dataset, args):
    """``(indices, selection)``: the examples --overfit trains on, and their record.

    ``head`` is the first N examples of the split, exactly as before the flag
    existed. ``stratified`` is every example of the records
    :func:`stratified_records` draws. Either way the indices are ascending, so the
    stop rule and the gate still read the examples back in dataset order.

    ``selection`` is what the run has to say about its draw: the mode, the ecg_ids in
    dataset order and the Yes/No counts per superclass. It is printed at startup and
    goes into every checkpoint's meta and overfit_final.pt's payload.
    """
    examples = dataset.examples
    if args.overfit_select == "stratified":
        chosen = set(stratified_records(examples, args.overfit // len(SUPERCLASSES),
                                        args.seed))
        indices = [i for i, e in enumerate(examples) if int(e.ecg_id) in chosen]
    else:
        indices = list(range(min(args.overfit, len(examples))))
    picked = [examples[i] for i in indices]
    return indices, {
        "select": args.overfit_select,
        "ecg_ids": list(dict.fromkeys(int(e.ecg_id) for e in picked)),
        "label_counts": label_counts(picked),
    }


def print_overfit_selection(selection, n_examples, seed):
    """The draw, said at startup: the ecg_ids and the five by two Yes/No table."""
    how = (f"stratified, seed {seed}" if selection["select"] == "stratified"
           else "head, the first examples of the train split")
    ids = selection["ecg_ids"]
    print(f"overfit selection: {how} -- {n_examples} examples from {len(ids)} records")
    print("  ecg_ids: " + " ".join(str(i) for i in ids))
    print(f"  {'superclass':<12} {'Yes':>5} {'No':>5}")
    for sc, n in selection["label_counts"].items():
        print(f"  {sc:<12} {n['yes']:>5} {n['no']:>5}")
    sys.stdout.flush()


def build_loaders(args, tokenizer, data_cfg, yes_no=None):
    """``(train_loader, val_loader, pos_weight)``. ``yes_no`` is ``(yes_ids, no_ids)``
    when the run has a linear head: the batches then carry consistency_pos, every
    target is checked against the consistency guards first, and ``pos_weight`` is
    :func:`superclass_pos_weights` over the train split (None without a head)."""
    from torch.utils.data import DataLoader, Subset

    splits = build_datasets(data_cfg)
    collate = make_collate_fn(tokenizer, max_length=args.max_length,
                              strip_think=data_cfg.strip_think)
    pos_weight = None
    if yes_no is not None:
        check_consistency_targets(tokenizer, {k: splits[k] for k in ("train", "val")},
                                  *yes_no)
        pos_weight = superclass_pos_weights(splits["train"].examples)
        print_pos_weights(splits["train"].examples, pos_weight)
        _RUN_META["linear_pos_weight"] = pos_weight
        collate = make_consistency_collate(tokenizer, collate, *yes_no)
    if args.loss_on == "decision":
        collate = make_decision_collate(tokenizer, collate)
    elif args.loss_on == "decision-no-context":
        collate = make_no_context_collate(tokenizer, collate)
    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.overfit is not None:
        try:
            indices, selection = select_overfit(splits["train"], args)
        except ValueError as err:
            sys.exit(f"--overfit-select {args.overfit_select}, FATAL: {err}")
        # A gate whose verdict depends on which examples it drew has to say which
        # ones: into the meta every checkpoint of this run carries.
        _RUN_META["overfit_selection"] = selection
        train = Subset(splits["train"], indices)
        print(f"OVERFIT mode: {len(indices)} train examples, no validation")
        print_overfit_selection(selection, len(indices), args.seed)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate,
                                  generator=g, drop_last=False)
        return train_loader, None, pos_weight

    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              generator=g, drop_last=True)
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    print(f"train {len(splits['train']):,} examples  val {len(splits['val']):,} examples")
    return train_loader, val_loader, pos_weight


# --------------------------------------------------------------------------- #
# Train / validate                                                            #
# --------------------------------------------------------------------------- #

def llm_binary_logits(logits, rows, positions, yes_ids, no_ids):
    """The LLM's log-odds of Yes at ``logits[rows, positions]``: logsumexp over the
    Yes ids minus logsumexp over the No ids. Its sigmoid is exactly
    evaluate.probability_from_logits, the P(Yes) every readout reports."""
    picked = logits[rows, positions].float()                  # (n, vocab)
    return (torch.logsumexp(picked[:, list(yes_ids)], dim=-1)
            - torch.logsumexp(picked[:, list(no_ids)], dim=-1))


def consistency_loss(llm_binary_logit, linear_logits):
    """BCE-with-logits of the LLM's Yes log-odds against sigmoid(linear logit),
    DETACHED: the head is the target, so nothing here may move it."""
    return F.binary_cross_entropy_with_logits(
        llm_binary_logit, torch.sigmoid(linear_logits).detach())


def assert_consistency_spares_head(consistency, head):
    """The reference's guard: d(consistency)/d(head) must be None for every head
    parameter. Raises GradFlowError naming the ones it reaches. Walks no node: the
    autograd engine only runs what leads to the inputs asked for, and none should."""
    params = list(head.named_parameters())
    grads = torch.autograd.grad(consistency, [p for _, p in params],
                                retain_graph=True, allow_unused=True)
    leaked = [name for (name, _), g in zip(params, grads) if g is not None]
    if leaked:
        raise GradFlowError(f"the consistency loss reaches the linear head's "
                            f"{leaked}: its target is not detached")


def check_head_gradients(head):
    """One-time, beside check_gradients: the linear loss reached every head parameter."""
    for name, p in head.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise GradFlowError(f"linear head {name}: grad is "
                                + ("None" if p.grad is None else "non-finite"))


def make_objective(args, head=None, yes_ids=None, no_ids=None, pos_weight=None):
    """``objective(model, resampler, batch, dev) -> (total, terms)``.

    ``total`` is what backward() is called on: lambda_llm * llm + lambda_linear *
    linear + lambda_consistency * consistency. ``terms`` holds each term unweighted
    as a float, for the log, with the mean |P_llm(Yes) - P_linear(Yes)| (``gap``)
    and how many rows the consistency term was over and sat out.

    Without a head it is the splice's loss times lambda_llm and nothing else, so a
    run with --lambda-linear 0 --lambda-consistency 0 takes the step it always took.

    llm is the --loss-on quantity, unchanged. linear is the head's BCE-with-logits
    on the resampler's own latents (not detached: it trains the resampler too) with
    the example's superclass pos_weight. consistency is read off the SAME forward's
    logits at the row before each gold Yes/No token (batch["consistency_pos"]), over
    the rows that have one; with none in the batch it is absent from the sum and
    the log. Whenever it is in the sum, assert_consistency_spares_head runs.
    """
    if head is not None and pos_weight is None:
        raise ValueError("a linear head needs the per-superclass pos_weight")

    def objective(model, resampler, batch, dev):
        if head is None:
            llm = splice_forward(model, resampler, batch, dev,
                                 task_stream=args.task_stream, loss_on=args.loss_on)
            return args.lambda_llm * llm, {"llm": float(llm.detach())}
        llm, latents, logits, Lq = splice_forward(
            model, resampler, batch, dev, task_stream=args.task_stream,
            loss_on=args.loss_on, return_aux=True)
        linear_logits = head(latents)            # outside autocast: fp32
        weights = torch.tensor([pos_weight[sc] for sc in batch["superclass"]],
                               dtype=torch.float32, device=linear_logits.device)
        linear = F.binary_cross_entropy_with_logits(
            linear_logits, batch["head_label"].to(linear_logits.device),
            pos_weight=weights)
        total = args.lambda_llm * llm + args.lambda_linear * linear
        pos = batch["consistency_pos"]
        rows = (pos >= 0).nonzero().flatten()
        terms = {"llm": float(llm.detach()), "linear": float(linear.detach()),
                 "consistency": None, "gap": None, "n_consistency": len(rows),
                 "n_skipped": len(pos) - len(rows)}
        if len(rows):
            llm_logit = llm_binary_logits(
                logits, rows.to(logits.device), (Lq + pos[rows] - 1).to(logits.device),
                yes_ids, no_ids).to(linear_logits.device)
            head_logit = linear_logits[rows.to(linear_logits.device)]
            consistency = consistency_loss(llm_logit, head_logit)
            if args.lambda_consistency > 0:
                if consistency.requires_grad:
                    assert_consistency_spares_head(consistency, head)
                total = total + args.lambda_consistency * consistency
            terms["consistency"] = float(consistency.detach())
            terms["gap"] = float((torch.sigmoid(llm_logit)
                                  - torch.sigmoid(head_logit)).abs().mean().detach())
        return total, terms

    return objective


def mean_terms(window):
    """The terms of a window of batches, reduced: llm and linear are means over the
    batches, consistency and gap means over the rows they were taken on."""
    out = {"llm": sum(t["llm"] for t in window) / max(len(window), 1)}
    if window and "linear" in window[0]:
        out["linear"] = sum(t["linear"] for t in window) / len(window)
        n = sum(t["n_consistency"] for t in window)
        out["n_consistency"] = n
        out["n_skipped"] = sum(t["n_skipped"] for t in window)
        for key in ("consistency", "gap"):
            out[key] = (sum(t[key] * t["n_consistency"] for t in window
                            if t[key] is not None) / n) if n else None
    return out


def describe_terms(means, args, loss_name):
    """One log fragment: the weighted total, then each term unweighted and weighted.

    Without a head and at lambda_llm 1 it is the bare ``{loss_name} X`` every log
    before the flags printed, so those logs still read the same."""
    if "linear" not in means and args.lambda_llm == 1.0:
        return f"{loss_name} {means['llm']:.4f}"
    parts, total = [], 0.0
    for name, key, lam in ((loss_name, "llm", args.lambda_llm),
                           ("linear", "linear", args.lambda_linear),
                           ("consistency", "consistency", args.lambda_consistency)):
        if key not in means:
            continue
        v = means[key]
        if v is None:
            parts.append(f"{name} n/a (no row reached its conclusion)")
            continue
        total += lam * v
        parts.append(f"{name} {v:.4f} (x{lam:g} = {lam * v:.4f})")
    line = f"total {total:.4f}  " + "  ".join(parts)
    if "gap" in means:
        gap = "n/a" if means["gap"] is None else f"{means['gap']:.4f}"
        line += (f"  |p_llm-p_linear| {gap} over {means['n_consistency']} rows, "
                 f"{means['n_skipped']} past --max-length")
    return line


@torch.no_grad()
def validate(model, resampler, val_loader, dev, objective, head=None):
    """:func:`mean_terms` over the validation split, everything in eval mode."""
    resampler.eval()
    if head is not None:
        head.eval()
    window = [objective(model, resampler, batch, dev)[1] for batch in val_loader]
    resampler.train()
    if head is not None:
        head.train()
    return mean_terms(window)


def _log_mem():
    return "  ".join(f"gpu{i} {p:.1f}GiB" for i, p in enumerate(gpu_peaks()))


@torch.no_grad()
def prefix_scale_line(model, resampler, batch, dev, task_stream: str = "prompt"):
    """Where the prefix actually sits, relative to the tokens it is spliced against.

    The LayerNorm gain is initialised to put the latents on the student's embedding
    scale (times --prefix-scale), but it is a trainable parameter: this measures
    where it has drifted to. Reports the mean per-position latent norm, the mean norm
    of the real prompt/target tokens, their ratio, and ||gain||, which is what the
    latent norm reduces to when the normalised vectors have unit variance -- so a
    gap between ||gain|| and the measured norm says the latents are not unit-variance
    going into the norm, and movement in ||gain|| alone is the optimiser pulling the
    prefix louder or quieter than it was initialised.

    Mirrors the resampler call in probe_memory.splice_forward; one extra resampler
    forward, no LLM forward, so it is cheap next to a training step.
    """
    ecg = batch["ecg_embed"].to(dev)
    input_ids = batch["input_ids"].to(dev)
    attn = batch["attention_mask"].to(dev)
    task_ids, task_mask = task_stream_ids(batch, task_stream)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok_embeds = model.get_input_embeddings()(input_ids)
        latents = resampler(ecg, task_embeddings=model.get_input_embeddings()(
            task_ids.to(dev)), task_mask=task_mask.to(dev))
    lat = float(latents.float().norm(dim=-1).mean())
    tok = float(tok_embeds.float().norm(dim=-1)[attn.bool()].mean())
    out_norm = getattr(resampler, "output_norm", None)
    gain = (f"  ||gain|| {float(out_norm.weight.detach().float().norm()):.3f}"
            if out_norm is not None else "")
    return (f"latents {lat:.3f}  tokens {tok:.3f}  ratio {lat / max(tok, 1e-9):.3f}"
            + gain)


def _by_label(means):
    return "  ".join(f"label={label} " + ("no examples" if means[label] is None
                                          else f"{means[label]:.4f}")
                     for label in (True, False))


def train_overfit(model, resampler, optimizer, loader, eval_batches, dev, args,
                  no_context_stop=None):
    """Train until the overfit set's eval-mode mean loss reaches the target, or the cap.

    The stop rule is gate condition 1's own quantity (overfit_gate.eval_mean_loss
    over ``eval_batches``, every --overfit-eval-every steps), not the loss of the
    batch just trained on: one easy batch in train mode says little about the mean
    over all of them in eval mode, and a loop that stops on the former can fail the
    gate the moment it stops.

    Reaching the target is still NOT the gate: run 1 reached 0.0496 and its free
    generation was degenerate. The caller runs the gate on the result either way, so
    a run that stops at the cap still reports all three conditions.

    Under either decision mode the same holds for the condition judged there. The
    step loss is the decision-token mean of the batch (with no target context under
    decision-no-context, where it is the stop rule's own quantity in train mode),
    and the stop rule is
    condition 4 (a)'s quantity (overfit_gate.eval_no_context_loss, through
    ``no_context_stop``): the eval-mode decision-token loss after the prompt and
    "Conclusion:" alone, which has to reach the target on label=True and on
    label=False separately, because that is how the gate judges it. A mean over
    both would stop a constant-Yes resampler on a set where positives dominate,
    for the gate to fail it at once. Condition 3's teacher-forced number
    (overfit_gate.eval_decision_loss over ``eval_batches``) is logged beside it at
    every check: it is not judged, and stopping on it let a run stop on a number
    the judged condition then failed. Every line is labelled decision-token, so
    none can be read against a --loss-on all log.

    ``no_context_stop`` is ``(tokenizer, dataset, indices)``, the overfit examples
    in the gate's order; required under DECISION_MODES, unused otherwise.
    ``eval_batches`` carries the FULL target under both, with the decision mask, so
    condition 3's line is the teacher-forced number in both.

    Returns the number of steps taken.
    """
    decision = args.loss_on in DECISION_MODES
    if decision and no_context_stop is None:
        raise ValueError(f"--loss-on {args.loss_on} stops on condition 4 (a), which "
                         "needs no_context_stop")
    loss_name = {"all": "loss", "decision": "decision-token loss",
                 "decision-no-context": "no-context decision-token loss"}[args.loss_on]
    stop_name = ("eval-mode no-context decision-token mean, worse label" if decision
                 else "eval-mode mean")
    n_examples = sum(len(b["ecg_id"]) for b in eval_batches)
    if decision:
        print(f"overfit [--loss-on {args.loss_on}]: train until the eval-mode "
              f"NO-CONTEXT DECISION-TOKEN mean (gate condition 4 (a): the prompt, "
              f"then \"Conclusion:\" alone) over all {n_examples} examples is <= "
              f"{args.overfit_target_loss} on label=True and on label=False "
              f"separately (checked every {args.overfit_eval_every} steps), at most "
              f"{args.overfit_max_steps} steps; then run the gate")
    else:
        print(f"overfit: train until the eval-mode mean loss over all {n_examples} "
              f"examples is <= {args.overfit_target_loss} (checked every "
              f"{args.overfit_eval_every} steps), at most {args.overfit_max_steps} "
              "steps; then run the gate")
    resampler.train()
    step = 0
    checked = False
    while step < args.overfit_max_steps:
        for batch in loader:
            t0 = time.time()
            loss = splice_forward(model, resampler, batch, dev,
                                  task_stream=args.task_stream, loss_on=args.loss_on)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not checked:                       # one-time grad-flow gate
                check_gradients(model, resampler)
                checked = True
            torch.nn.utils.clip_grad_norm_(resampler.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            lv = float(loss.detach())
            print(f"step {step:5d}  {loss_name} {lv:.4f}  {time.time()-t0:.2f}s  "
                  f"{_log_mem()}", flush=True)
            if step == 1 or step % args.prefix_log_every == 0:
                print(f"    prefix scale @ step {step}: "
                      f"{prefix_scale_line(model, resampler, batch, dev, args.task_stream)}",
                      flush=True)
            if step % args.overfit_eval_every == 0:
                if decision:
                    tok, ds, idx = no_context_stop
                    nc = eval_no_context_loss(model, resampler, tok, ds, idx, dev, args)
                    tf = eval_decision_loss(model, resampler, eval_batches, dev,
                                            args.task_stream)
                    # A label with no examples cannot hold the loop open; the gate
                    # fails condition 4 (a) on it whatever the other label reaches.
                    mean = max(v for v in (nc[True], nc[False]) if v is not None)
                    print(f"    eval-mode no-context decision-token mean over all "
                          f"{n_examples} examples @ step {step}: {nc['all']:.4f}  "
                          f"{_by_label(nc)}  (stop when each label is <= "
                          f"{args.overfit_target_loss})", flush=True)
                    print(f"    eval-mode teacher-forced decision-token mean @ step "
                          f"{step}: {tf['all']:.4f}  {_by_label(tf)}  (condition 3, "
                          "not judged and not the stop rule)", flush=True)
                else:
                    mean = eval_mean_loss(model, resampler, eval_batches, dev,
                                          args.task_stream)
                    print(f"    eval-mode mean over all {n_examples} examples @ step "
                          f"{step}: {mean:.4f}  (stop at <= {args.overfit_target_loss})",
                          flush=True)
                if mean <= args.overfit_target_loss:
                    print(f"overfit stopped: {stop_name} {mean:.4f} <= "
                          f"{args.overfit_target_loss} at step {step}")
                    return step
            if step >= args.overfit_max_steps:
                break
    print(f"overfit stopped: {stop_name} did NOT reach {args.overfit_target_loss} "
          f"in {step} steps; the gate runs regardless")
    return step


def train(model, resampler, optimizer, train_loader, val_loader, dev, args,
          start_epoch, global_step, best_val, objective=None, head=None):
    # Under either decision mode every loss here is the decision-token mean, and
    # each line says which: it starts near 0.5 where the whole-target loss starts
    # near 3, and the two decision modes differ in what precedes the token.
    loss_name = {"all": "loss", "decision": "decision-token loss",
                 "decision-no-context": "no-context decision-token loss"}[args.loss_on]
    tag = {"all": "", "decision": "decision_token_",
           "decision-no-context": "no_context_decision_token_"}[args.loss_on]
    objective = objective or make_objective(args)
    params = list(resampler.parameters())
    for epoch in range(start_epoch, args.epochs):
        resampler.train()
        if head is not None:
            head.train()
        gpu_reset()
        window_t0 = time.time()
        running = []
        for batch in train_loader:
            loss, terms = objective(model, resampler, batch, dev)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step == 0:                  # one-time grad-flow gate
                check_gradients(model, resampler)
                if head is not None:
                    check_head_gradients(head)
            # Clipped apart: the head's gradient, at 10x the learning rate, does not
            # spend the resampler's clip budget.
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if head is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            running.append(terms)

            if global_step % args.log_every == 0:
                dt = (time.time() - window_t0) / args.log_every
                print(f"epoch {epoch} step {global_step} "
                      f"{describe_terms(mean_terms(running[-args.log_every:]), args, loss_name)} "
                      f"{dt*1000:.0f}ms/step  {_log_mem()}", flush=True)
                gpu_reset()
                window_t0 = time.time()

            if global_step % args.ckpt_every == 0:
                save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch, global_step,
                                best_val, is_best=False, head=head)

            if _PREEMPT["flag"]:
                print(f"preemption signal received at step {global_step}; saving "
                      "emergency checkpoint and requeueing", flush=True)
                save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch, global_step,
                                best_val, is_best=False, head=head)
                requeue_and_exit()

        val = validate(model, resampler, val_loader, dev, objective, head)
        train_means = mean_terms(running)
        # best.pt stays the lowest validation LLM loss, as before the other terms.
        val_loss = val["llm"]
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        print(f"== epoch {epoch} done  train_{tag}loss "
              f"{train_means['llm']:.4f}  val_{tag}loss {val_loss:.4f}"
              f"{'  (best)' if is_best else ''}", flush=True)
        if head is not None:
            print(f"   train: {describe_terms(train_means, args, loss_name)}\n"
                  f"   val:   {describe_terms(val, args, loss_name)}", flush=True)
        save_checkpoint(args.ckpt_dir, resampler, optimizer, epoch + 1, global_step,
                        best_val, is_best=is_best, head=head)
        if args.save_every_epoch:
            path = save_epoch_checkpoint(args.ckpt_dir, resampler, epoch + 1,
                                         global_step, best_val, head=head)
            print(f"   epoch checkpoint: {path}", flush=True)


def main():
    args = parse_args()
    log_git_state()                           # first line of the log: what code ran
    if args.lr is None:                       # 1e-4 in the gate, 1e-5 for real training
        args.lr = 1e-4 if args.overfit is not None else 1e-5
    _RUN_META.update({
        "prefix_scale": args.prefix_scale,
        "task_stream": args.task_stream,
        "loss_on": args.loss_on,
        "input_layer_norm": not args.no_input_norm,
        "final_output_norm": not args.no_output_norm,
        "strip_think": bool(args.strip_think),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "overfit": args.overfit,
        "seed": args.seed,
        # what the objective was, next to task_stream and strip_think: evaluation
        # loads the head by these, and a resume must match them
        **objective_lambdas(args),
        "linear_head_lr": args.linear_head_lr if args.lambda_linear > 0 else None,
    })
    set_seed(args.seed)
    _install_signal_handlers()

    cfg_kwargs = {"strip_think": args.strip_think}
    if args.emb_cache:
        cfg_kwargs["emb_cache_dir"] = args.emb_cache
    if args.manifest:
        cfg_kwargs["manifest_path"] = args.manifest
    if args.teacher:
        cfg_kwargs["teacher_path"] = args.teacher
    data_cfg = DatasetConfig(**cfg_kwargs)
    print(f"prompt: strip_think={data_cfg.strip_think}")
    if args.loss_on == "decision":
        print(f"loss on: decision -- ONE target position per example, the Yes/No token "
              f"after {MARKER!r}. Every training, stop-rule and validation loss this "
              "run logs is the mean of that token's loss and is labelled "
              "decision-token; none of them is comparable with a --loss-on all log.")
    elif args.loss_on == "decision-no-context":
        print(f"loss on: decision-no-context -- ONE position per example, the Yes/No "
              f"token, with each target cut to {MARKER!r} and that token: no Evidence, "
              "no Reasoning in context, as gate condition 4 asks for it. Every loss "
              "this run logs is that token's mean, labelled no-context decision-token; "
              "none is comparable with a --loss-on all or decision log.")

    preflight(args, data_cfg)
    # A resume continues the checkpoint's run, so it has to continue it on the same
    # prompt. Checked here, before the student loads; there is no way through, since
    # switching prompt format mid-run is not an experiment anyone can read.
    if args.overfit is None and _rotating(args.ckpt_dir):
        require_strip_think_match(_rotating(args.ckpt_dir)[-1], args.strip_think,
                                  escape_flag=None)
        require_task_stream_match(_rotating(args.ckpt_dir)[-1], args.task_stream)
        require_objective_match(_rotating(args.ckpt_dir)[-1],
                                *checkpoint_objective(_rotating(args.ckpt_dir)[-1]),
                                objective_lambdas(args))
    print(f"resampler task stream: {args.task_stream}")
    print("objective: " + "  ".join(f"{k} {v:g}" for k, v in
                                    objective_lambdas(args).items())
          + (f"  linear head lr {args.linear_head_lr:g}" if args.lambda_linear > 0
             else "  (no linear head)"))

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible; training needs GPUs")
    print(f"visible GPUs: {torch.cuda.device_count()}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    yes_no = yes_no_token_ids(tokenizer) if args.lambda_linear > 0 else None
    train_loader, val_loader, pos_weight = build_loaders(args, tokenizer, data_cfg,
                                                         yes_no)

    print("loading frozen student (bf16, device_map=auto, use_cache=False) ...")
    model = load_student(args.model_dir)
    if args.grad_checkpointing == "on":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dev = next(model.get_input_embeddings().parameters()).device

    resampler = Resampler(PerceiverResamplerConfig.ecg(
        input_layer_norm=not args.no_input_norm,
        final_output_norm=not args.no_output_norm,
    )).to(dev)
    print(f"resampler architecture: input_layer_norm={not args.no_input_norm} "
          f"final_output_norm={not args.no_output_norm}")
    # Start the latents at the student's own embedding scale. The splice puts them
    # in front of real token embeddings, and attention logits are dot products, so
    # a prefix far shorter than those embeddings draws almost no attention weight
    # and the frozen LLM reads past the ECG. This only sets where training starts:
    # the LayerNorm gain trains like any other parameter, and on resume the
    # checkpoint's learned value replaces it (resume() loads after this).
    embed_norm = mean_embedding_norm(model.get_input_embeddings().weight)
    if args.no_output_norm:
        print(f"student mean embedding norm: {embed_norm:.4f}; output LayerNorm is OFF "
              "(--no-output-norm), so there is no gain to calibrate: the latent scale "
              "is whatever the block stack produces")
    else:
        target = args.prefix_scale * embed_norm
        gain = resampler.calibrate_output_norm(target)
        print(f"student mean embedding norm: {embed_norm:.4f}; prefix_scale "
              f"{args.prefix_scale:g} -> target latent norm {target:.4f}; output "
              f"LayerNorm gain set to {gain:.6f} "
              f"(= {target:.4f} / sqrt({resampler.config.embed_dim}))")
    resampler.train()
    n_train = sum(p.numel() for p in resampler.parameters() if p.requires_grad)
    n_frozen_grad = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"resampler trainable params: {n_train:,} (fp32) on {dev}; "
          f"LLM params with requires_grad: {n_frozen_grad} (must be 0)")

    head = None
    groups = [{"params": resampler.parameters(), "lr": args.lr}]
    if args.lambda_linear > 0:
        head = AuxiliaryLinearHead(resampler.config.embed_dim).to(dev).train()
        groups.append({"params": head.parameters(), "lr": args.linear_head_lr})
        print(f"linear head: LayerNorm + Linear({resampler.config.embed_dim}, 1) on the "
              f"mean-pooled latents, "
              f"{sum(p.numel() for p in head.parameters()):,} params, lr "
              f"{args.linear_head_lr:g}")
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f"AdamW lr={args.lr} weight_decay={args.weight_decay} "
          f"grad_clip={args.grad_clip} batch={args.batch_size} "
          f"grad_checkpointing={args.grad_checkpointing}")

    if args.overfit is not None:
        # The stop rule and the gate both read the overfit examples back in dataset
        # order, through the same batches.
        overfit_set = train_loader.dataset          # the Subset build_loaders made
        dataset, indices = overfit_set.dataset, list(overfit_set.indices)
        # Materialised through the loader's own collate, so every overfit example
        # has had its decision token placed (and, under decision-no-context, its
        # target cut), or the run has stopped, before the first step.
        eval_batches = gate_batches(tokenizer, dataset, indices, args,
                                    collate=train_loader.collate_fn)
        if args.loss_on == "decision-no-context":
            # Condition 3's log line reads the token with the FULL target in context,
            # which these cut batches no longer hold.
            eval_batches = gate_batches(
                tokenizer, dataset, indices, args, collate=make_decision_collate(
                    tokenizer, make_collate_fn(tokenizer, max_length=args.max_length,
                                               strip_think=dataset.config.strip_think)))
        steps = train_overfit(model, resampler, optimizer, train_loader, eval_batches,
                              dev, args, no_context_stop=(tokenizer, dataset, indices))
        # Drop the last step's gradients first: nothing below needs them.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        ok = run_gate(model, resampler, tokenizer, dataset, indices, args)
        # After the gate, so the file carries the verdict on the weights it holds.
        # A write that fails (scratch quota) is said on the closing line and does not
        # touch the exit code: that is the gate's verdict, and an uncaught error here
        # would exit 1 on a gate that passed.
        # None is condition 4 not judged (too few donor-answers-opposite examples):
        # not a pass, so it exits 1 and is saved as gate_passed False, but it is
        # not called a failure either.
        verdict = {True: "gate PASSED", False: "gate FAILED", None: "gate NOT JUDGED"}[ok]
        try:
            path = save_overfit_final(args.ckpt_dir, resampler,
                                      steps // len(train_loader), steps, ok is True,
                                      args)
        except (OSError, RuntimeError) as err:
            print(f"{verdict}; the resampler it judged (step {steps}) was NOT saved to "
                  f"{os.path.join(args.ckpt_dir, OVERFIT_FINAL)}: {err}", flush=True)
        else:
            print(f"{verdict}; the resampler it judged (step {steps}) is in {path}",
                  flush=True)
        sys.exit(0 if ok is True else 1)

    start_epoch, global_step, best_val = resume(args.ckpt_dir, resampler, optimizer,
                                                dev, args.loss_on, head=head,
                                                lambdas=objective_lambdas(args))
    objective = make_objective(args, head, *(yes_no or (None, None)), pos_weight)
    train(model, resampler, optimizer, train_loader, val_loader, dev, args,
          start_epoch, global_step, best_val, objective, head)


if __name__ == "__main__":
    try:
        main()
    except DecisionTokenError as err:
        sys.exit(f"--loss-on {_RUN_META.get('loss_on', 'decision')}, FATAL: {err}")
    except ConsistencyTargetError as err:
        sys.exit(f"--lambda-linear {_RUN_META.get('lambda_linear')}, FATAL: {err}")
