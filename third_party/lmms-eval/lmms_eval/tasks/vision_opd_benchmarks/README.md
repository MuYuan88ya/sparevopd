# Vision-OPD benchmark additions

This directory adds the two benchmarks used by Vision-OPD that are absent
from the pinned upstream lmms-eval revision.

## ZoomBench

Task name: `zoombench`.

The task loads `inclusionAI/ZoomBench` and sends the full `image` to the model.
The dataset's `crop_image` is intentionally ignored, matching the existing
Vision-OPD evaluation. `query` is the prompt and `response` is the reference
answer.

## VisualProbe

Group name: `visualprobe`.

The group contains `visualprobe_easy`, `visualprobe_medium`, and
`visualprobe_hard`, loaded from the corresponding `Mini-o3/VisualProbe_*`
repositories. The group metric is the unweighted mean of the three difficulty
accuracies, matching the legacy Vision-OPD metric.

## Judge

Both benchmarks first try normalized exact matching and MathRuler when it is
installed. Remaining answers use the same semantic-equivalence Yes/No prompt
as `eval/judge_qwenlm.py`.

Judge output is parsed like HRBench's option output: formatting such as
`Yes.`, `(NO)`, `The answer is yes`, and `<answer>No</answer>` is accepted only
when it contains one unambiguous decision. Empty, undecidable, or conflicting
Yes/No responses are retried instead of being silently scored as incorrect.
The final failed attempt raises an evaluation error.

The staged Vision-OPD wrapper supplies Judge settings through
`VISION_OPD_JUDGE_*` variables and persists successful per-sample results in
the experiment's `judge_cache` directory. The cache records the Judge parsing
protocol version and is reused only when the question, target, prediction, and
Judge model are unchanged. Caches from older parsing protocols are ignored.
