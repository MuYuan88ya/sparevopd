# Vendored lmms-eval

- Upstream: <https://github.com/EvolvingLMMs-Lab/lmms-eval>
- Commit: `84770436f0c3b099efbb56f75c6c0db79bb91739`
- Upstream package version: `0.7.2`

The source is vendored instead of included as a Git submodule so an evaluation
checkout is self-contained and does not need a separate lmms-eval installation
step.

Vision-OPD carries one focused compatibility patch in
`lmms_eval/models/simple/from_log.py`: the `from_log` backend can strictly
replay the per-task JSONL files emitted by the current `--predict_only` flow.
This provides a hard generation/evaluation phase boundary. The evaluation
phase cannot silently fall back to the generation API.

To refresh the vendor tree, copy a reviewed upstream revision into this
directory without its `.git` directory, restore the local `from_log` patch and
tests, update the commit above, and run the validation documented in
`eval_lmms/README.md`.
