"""Teacher input pipeline (Phase 1.4).

Turns a manifest entry into the exact prompt the privileged teacher receives:

    report.py   clean the free-text report (pure, no I/O)
    view.py     assemble the teacher's view of a record (deterministic, serialisable)
    prompt.py   render a view + superclass question into the final prompt string

No module here calls any API. Generation is a later, separate step.
"""
