# mcq-length-floor

How well does **"pick the longest option"** do on a multiple-choice benchmark, without reading the question? And
how does that change slice by slice?

*Built by AI agents working in the name of MV2 of Munim, Inc. Apache-2.0.*

A closed-book score is only meaningful against the floor a question-free heuristic already reaches. That floor is
often well above chance and varies a lot between slices of one benchmark. The
[eval-reliability-audits](https://github.com/munimtechnologies/munimtechnologies.github.io/tree/main/eval-reliability-audits)
catalogue found:

- on MMLU-Pro, picking the longest option scores 18.38% against 11.09% chance (1.66x), more than MMLU's 1.12x;
- on MMLU-Pro's STEM-website (`src` `stemez-`) biology and psychology questions it scores about 67%;
- ten of 45 published model runs score below that heuristic on those 656 questions;
- the shortcut survives translation: on MMLU-ProX it beats chance by about the same margin in all 29 languages.

`mcq-length-floor` computes the floor for your own data in one command.

## What it reports

For every slice (all rows, each value of each `--group` column, and optionally each slice split by negatively worded
stems):

| column | meaning |
| --- | --- |
| n | questions in the slice |
| chance | exact chance baseline, the mean of 1/k over the slice's questions (k can vary) |
| pick-longest | tie-aware accuracy of picking the longest option (credit 1/t when the gold is one of t tied longest) |
| pick-shortest | the same for the shortest option |
| lift | pick-longest minus chance |
| ratio | pick-longest / chance |
| z | lift divided by its standard error over questions (`-` when undefined) |

Lengths are Unicode characters after trimming whitespace (`--unit words` counts words instead).

## Use

```
pip install "mcq-length-floor[hf] @ git+https://github.com/munimtechnologies/mcq-length-floor"

# a HuggingFace dataset, read through the datasets-server parquet API (needs pyarrow)
mcq-length-floor --hf TIGER-Lab/MMLU-Pro --config default --split test \
    --options options --answer answer_index --group src

# separate option columns, a subject slice, and NOT / EXCEPT stems split out
mcq-length-floor --hf openlifescienceai/medmcqa --config default --split validation \
    --option-cols opa,opb,opc,opd --answer cop --group subject_name --polarity --question question

# a local file (JSONL or CSV), JSON output
mcq-length-floor --jsonl my_benchmark.jsonl --options choices --answer label --json --out floor.json
```

Input rules:

- **Options**: a list column (`--options`, also a JSON-encoded list or a dict, read in key order) or several columns
  (`--option-cols a,b,c,d`). Blank options and any `--pad-token` (for example `N/A`) are dropped; a row whose gold
  option is blank is skipped.
- **Answer**: an integer index (0-based unless `--one-based`), a single letter A-Z, or the exact text of one option.
  Integers are tried first, then letters, then text.
- **Polarity** (`--polarity`): a stem is "negative" when its last sentence contains not, except, false, incorrect,
  untrue, least or n't as a whole word. Only the last sentence is read, so a negation inside a case vignette does not
  count. It is a simple rule, not a parser; check a sample on your data. Polarity slices are named
  `<slice> | stem=neg` / `stem=pos`; in the markdown table the `|` is escaped so the columns stay aligned.
- Skipped rows are counted by reason under the table.

## Validated

Before this release the tool was run in a sealed container (job `mcq-length-floor-validate-v1`, 30 October 2026,
`python:3.12`, unit tests then the two commands above) and checked against the catalogue's independently computed
numbers:

| check | audit figure | tool output |
| --- | --- | --- |
| MMLU-Pro test, all 12,032 questions: pick-longest | 18.38% (audit 61) | 0.183768 |
| MMLU-Pro test: chance (mean 1/k) | 11.09% (audit 61) | 0.110877 |
| `src=stemez-Biology`, n 351: pick-longest | 0.6670736670736671 (audit 63) | 0.6670736670736671 |
| `src=stemez-Psychology`, n 305: pick-longest | 0.6700858704137392 (audit 63) | 0.6700858704137392 |

The per-source figures match to every printed digit. On the same run MedMCQA's validation split (4,183 questions,
4 options) gave pick-longest 0.2999 against 0.25 chance (1.20x, z 7.6), with almost the same lift on the 754
negatively worded stems (+0.0498) as on the other 3,429 (+0.0499). That run also exposed a table bug, since fixed:
the polarity separator `|` split the slice cell in two. JSON output was not affected.

## How to read it

- A slice with a ratio well above 1 and a large z leaks the answer through length. Compare model scores on that
  slice with its pick-longest figure, not with a flat 1/k.
- A negative lift with a high pick-shortest means the keyed answers are systematically short (common when answers
  are numbers or single terms).
- The tool measures how the options were written. It does not show that any model uses the cue; for that, see audits
  65 and 66 in the catalogue.

## Tests

```
python -m unittest -v
```

The core is standard library only and every figure is computed exactly (`fractions.Fraction`) before it is turned
into a float. The unit tests are hand-traced fixtures.

## Licence

Apache-2.0. Datasets you run it on keep their own licences.
