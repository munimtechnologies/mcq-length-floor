"""mcq-length-floor: how well does "pick the longest option" do on a multiple-choice benchmark, slice by slice?

A question-free heuristic that always picks the longest (or the shortest) option is a floor every closed-book score
should be compared with. On MMLU-Pro it scores 18.4% against 11.1% chance overall and about 67% on the STEM-website
biology and psychology questions (eval-reliability-audits 61-63, MV2 of Munim, Inc.). This tool computes that floor
for any multiple-choice data: a JSONL or CSV file, or a HuggingFace dataset read through the datasets-server parquet
API.

For each slice (all rows, then each value of every --group column, optionally split by negatively worded stems) it
reports n, the exact chance baseline (mean 1/k), the tie-aware accuracy of pick-longest and pick-shortest, the lift of
pick-longest over chance, its standard error over questions and z = lift / se.

Usage examples:

    python mcq_length_floor.py --jsonl data.jsonl --options options --answer answer_index --group category
    python mcq_length_floor.py --hf TIGER-Lab/MMLU-Pro --split test --options options --answer answer_index \\
        --group src --pad-token N/A
    python mcq_length_floor.py --hf openlifescienceai/medmcqa --split validation --option-cols opa,opb,opc,opd \\
        --answer cop --group subject_name --polarity --question question

The core is standard library only; --hf needs pyarrow. Apache-2.0. Authored by AI agents working in the name of
MV2, Munim, Inc.
"""

import argparse
import csv
import io
import json
import math
import re
import sys
import urllib.request
from fractions import Fraction

__version__ = "0.1.0"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NEG_RE = re.compile(r"\b(not|except|false|incorrect|untrue|least)\b|n['’]t\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset=%s"
UA = {"User-Agent": "mcq-length-floor/%s (+https://github.com/munimtechnologies/mcq-length-floor)" % __version__}


# ----- lengths and credits -----

def length(option, unit="chars"):
    s = str(option).strip()
    return len(WORD_RE.findall(s)) if unit == "words" else len(s)


def credits(options, gold, unit="chars"):
    """(pick-longest credit, pick-shortest credit) for one question, each tie-aware (1/t when the gold is one of t
    tied extremes)."""
    L = [length(o, unit) for o in options]
    hi, lo = max(L), min(L)
    cl = Fraction(1, L.count(hi)) if L[gold] == hi else Fraction(0)
    cs = Fraction(1, L.count(lo)) if L[gold] == lo else Fraction(0)
    return cl, cs


def polarity(question):
    """'neg' when the question's last sentence holds not / except / false / incorrect / untrue / least / n't."""
    s = str(question or "").strip()
    parts = [p for p in SENT_SPLIT.split(s) if p.strip()]
    return "neg" if parts and NEG_RE.search(parts[-1]) else "pos"


# ----- turning a record into (options, gold) -----

def _blank(o, pad_tokens):
    return o is None or str(o).strip() == "" or str(o).strip() in pad_tokens


def extract(rec, options_col=None, option_cols=None, answer_col="answer", one_based=False, pad_tokens=()):
    """Return (options, gold_index, None) or (None, None, reason). Options may be a list column, a dict column
    (values in key order) or several columns; blank and pad-token options are dropped. The answer may be an index
    (0-based unless one_based), a letter, or the text of an option."""
    if option_cols:
        raw = [rec.get(c) for c in option_cols]
    else:
        raw = rec.get(options_col)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, None, "options_not_a_list"
        if isinstance(raw, dict):
            raw = [raw[k] for k in sorted(raw)]
    if not isinstance(raw, (list, tuple)):
        return None, None, "options_not_a_list"
    ans = rec.get(answer_col)
    gold_raw = None
    if isinstance(ans, bool):
        return None, None, "bad_answer"
    if isinstance(ans, int) or (isinstance(ans, str) and ans.strip().lstrip("-").isdigit()):
        gold_raw = int(ans) - (1 if one_based else 0)
    elif isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().upper() in LETTERS:
        gold_raw = LETTERS.index(ans.strip().upper())
    elif isinstance(ans, str):
        hits = [i for i, o in enumerate(raw) if o is not None and str(o).strip() == ans.strip()]
        if len(hits) == 1:
            gold_raw = hits[0]
    if gold_raw is None or not 0 <= gold_raw < len(raw):
        return None, None, "bad_answer"
    if _blank(raw[gold_raw], pad_tokens):
        return None, None, "gold_blank"
    keep = [i for i, o in enumerate(raw) if not _blank(o, pad_tokens)]
    if len(keep) < 2:
        return None, None, "fewer_than_2_options"
    return [raw[i] for i in keep], keep.index(gold_raw), None


# ----- accumulation -----

def new_acc():
    return {"n": 0, "pl": Fraction(0), "ps": Fraction(0), "chance": Fraction(0), "d_sq": Fraction(0)}


def add(acc, options, gold, unit="chars"):
    cl, cs = credits(options, gold, unit)
    ch = Fraction(1, len(options))
    acc["n"] += 1
    acc["pl"] += cl
    acc["ps"] += cs
    acc["chance"] += ch
    acc["d_sq"] += (cl - ch) ** 2


def report(acc):
    n = acc["n"]
    if not n:
        return {"n": 0}
    pl, ps, ch = acc["pl"] / n, acc["ps"] / n, acc["chance"] / n
    lift = pl - ch
    se = z = None
    if n > 1:
        var = (acc["d_sq"] - n * lift * lift) / (n - 1)
        se = math.sqrt(float(var) / n)
        z = float(lift) / se if se else None
    return {"n": n, "chance": float(ch), "pick_longest": float(pl), "pick_shortest": float(ps),
            "lift": float(lift), "ratio": float(pl / ch), "se": se, "z": z}


def census(records, options_col=None, option_cols=None, answer_col="answer", groups=(), question_col=None,
           split_polarity=False, unit="chars", one_based=False, pad_tokens=()):
    """Accumulate every record into slices. Returns (reports by slice name, skipped counts by reason)."""
    accs, skipped = {}, {}
    for rec in records:
        options, gold, why = extract(rec, options_col, option_cols, answer_col, one_based, pad_tokens)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        names = ["ALL"] + ["%s=%s" % (g, rec.get(g)) for g in groups]
        if split_polarity:
            p = "stem=" + polarity(rec.get(question_col))
            names = names + [nm + " | " + p for nm in names]
        for nm in names:
            add(accs.setdefault(nm, new_acc()), options, gold, unit)
    order = sorted(accs, key=lambda s: (s != "ALL", s.split(" | ")[0] != "ALL", s))
    return {s: report(accs[s]) for s in order}, skipped


# ----- input -----

def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def read_hf(dataset, config, split, columns):
    import pyarrow.parquet as pq  # only needed for --hf

    files = json.loads(_get(PARQUET_API % dataset).decode("utf-8"))["parquet_files"]
    urls = sorted(f["url"] for f in files if f["split"] == split and (config is None or f["config"] == config))
    if not urls:
        raise SystemExit("no parquet files for %s config=%s split=%s" % (dataset, config, split))
    for url in urls:
        table = pq.read_table(io.BytesIO(_get(url)), columns=columns)
        yield from table.to_pylist()


# ----- output -----

def markdown(reports, skipped):
    head = ("| slice | n | chance | pick-longest | pick-shortest | lift | ratio | z |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = [head]
    for s, r in reports.items():
        z = "%.1f" % r["z"] if r.get("z") is not None else "-"
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %+.4f | %.2f | %s |"
                     % (s, r["n"], r["chance"], r["pick_longest"], r["pick_shortest"], r["lift"], r["ratio"], z))
    if skipped:
        lines.append("")
        lines.append("Skipped: " + ", ".join("%s %d" % kv for kv in sorted(skipped.items())))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl")
    src.add_argument("--csv")
    src.add_argument("--hf", help="HuggingFace dataset id, read via the datasets-server parquet API (needs pyarrow)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    opt = ap.add_mutually_exclusive_group(required=True)
    opt.add_argument("--options", help="column holding a list (or dict) of options")
    opt.add_argument("--option-cols", help="comma-separated option columns, in order")
    ap.add_argument("--answer", required=True, help="answer column: index, letter or option text")
    ap.add_argument("--one-based", action="store_true", help="integer answers start at 1")
    ap.add_argument("--group", action="append", default=[], help="slice by this column (repeatable)")
    ap.add_argument("--polarity", action="store_true", help="also split every slice by negatively worded stems")
    ap.add_argument("--question", default="question", help="question column for --polarity")
    ap.add_argument("--pad-token", action="append", default=[], help="option text meaning 'no option' (e.g. N/A)")
    ap.add_argument("--unit", choices=["chars", "words"], default="chars")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a markdown table")
    ap.add_argument("--out", help="write the output to this file instead of stdout")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    option_cols = a.option_cols.split(",") if a.option_cols else None
    if a.jsonl:
        records = read_jsonl(a.jsonl)
    elif a.csv:
        records = read_csv(a.csv)
    else:
        cols = (option_cols or [a.options]) + [a.answer] + a.group + ([a.question] if a.polarity else [])
        records = read_hf(a.hf, a.config, a.split, list(dict.fromkeys(cols)))
    reports, skipped = census(records, a.options, option_cols, a.answer, a.group, a.question, a.polarity, a.unit,
                              a.one_based, tuple(a.pad_token))
    text = (json.dumps({"reports": reports, "skipped": skipped}, indent=1) if a.json
            else markdown(reports, skipped)) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
