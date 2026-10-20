"""Where the Yes/No answer sits in a target: the first token after "Conclusion:".

The one locator every script uses -- training's --loss-on decision masks, the overfit
gate's conditions 3 and 4, the sweep's no-context score and yn_token_loss.py's
report -- so none of them can disagree on which token carries the answer. It imports
nothing from the project, so any module can import it without a cycle.
"""

MARKER = "Conclusion:"


def tokenise_target(tokenizer, text):
    """Target ids exactly as make_collate_fn builds them, plus each token's char span."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc.input_ids)
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return ids, enc.offset_mapping


def decision_index(text, offsets):
    """Index of the first token after the marker, or (None, why) if it is ambiguous."""
    if text.count(MARKER) != 1:
        return None, f"{text.count(MARKER)} occurrences of {MARKER!r}"
    end = text.index(MARKER) + len(MARKER)
    for k, (s, e) in enumerate(offsets):
        if e > end:
            if s < end:
                return None, "a token straddles the end of the marker"
            return k, None
    return None, "no token after the marker"
