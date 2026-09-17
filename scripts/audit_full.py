import json, re, collections

PATH = "data/teacher/full_v6.jsonl"
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

LEAK = {
    "the report":          r"\bthe report\b",
    "is reported":         r"\bis reported\b|\bare reported\b",
    "annotat":             r"annotat",
    "known/required answer": r"known answer|required answer|correct answer|answer is affirmative|known diagnosis|answer must reflect|the required",
    "not described":       r"not described|no description",
    "diagnostic statement":r"diagnostic statement",
    "likelihood":          r"likelihood",
    "diagnostic finding":  r"diagnostic finding",
    "form statement":      r"form statement",
    "written interpretation": r"written (?:report|interpretation)|not captured in",
    "cardiologist":        r"cardiologist",
}

n = 0
parse_fail = []
empty_reason = []
mismatch = []
strip = collections.Counter()
leak_blocks = collections.Counter()
leak_records = collections.Counter()
blocks_total = 0
blocks_with_any_leak = 0
recs_with_any_leak = 0
diag_items = 0
lead_type = 0
yes_counts = collections.Counter()
completion = []

for line in open(PATH):
    r = json.loads(line); n += 1
    completion.append(r["usage"]["completion_tokens"])
    if r["label_mismatches"]: mismatch.append(r["ecg_id"])
    s = r.get("strip") or {}
    for k, v in s.items():
        if isinstance(v, int): strip[k] += v
    pc = r.get("parsed_clean") or r.get("parsed")
    if not pc.get("ok"): parse_fail.append(r["ecg_id"])
    rec_leaked = False
    for sc in SUPER:
        b = (pc.get("blocks") or {}).get(sc)
        if not b: continue
        blocks_total += 1
        rb = b.get("raw_block") or ""
        if b.get("conclusion") is True: yes_counts[sc] += 1
        if not (b.get("reasoning") or "").strip(): empty_reason.append((r["ecg_id"], sc))
        hit = False
        for name, pat in LEAK.items():
            if re.search(pat, rb, re.I):
                leak_blocks[name] += 1; hit = True
        if hit:
            blocks_with_any_leak += 1; rec_leaked = True
    if rec_leaked: recs_with_any_leak += 1
    diag_items += r.get("diagnosis_evidence_items", 0)
    lead_type += r.get("lead_type_blocks", 0)

print(f"records                {n}")
print(f"blocks                 {blocks_total}  (expected {n*5})")
print(f"label mismatches       {len(mismatch)}")
print(f"parse failures         {len(parse_fail)}  {parse_fail[:10]}")
print(f"empty Reasoning        {len(empty_reason)}  {empty_reason[:10]}")
print(f"mean completion tokens {sum(completion)/len(completion):.0f}")
print(f"strip totals           {dict(strip)}")
print(f"lead_type_blocks       {lead_type}")
print(f"diagnosis_evidence     {diag_items}  ({diag_items/n:.2f}/record)")
print()
print(f"blocks with any leak   {blocks_with_any_leak}/{blocks_total} "
      f"({100*blocks_with_any_leak/blocks_total:.1f}%)")
print(f"records with any leak  {recs_with_any_leak}/{n} ({100*recs_with_any_leak/n:.1f}%)")
print("by pattern (post-strip, in parsed_clean):")
for k, v in leak_blocks.most_common():
    print(f"   {k:<24} {v:>6}  ({100*v/blocks_total:.1f}% of blocks)")
print()
print("Yes blocks by superclass:", dict(yes_counts))