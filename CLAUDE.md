# gazekit — project rules

- Read `data/journal.jsonl` (`gazekit journal`) before assuming what has or
  hasn't been run; every CLI invocation is recorded there.
- Any new collection mode MUST implement `docs/COLLECTION_STANDARD.md`
  (pre-flight gate → sample gating → provenance → validation+verdict →
  journal registration). No blind collection.
- Never bypass the deploy gates (calibrate / train-cnn / iterate) by saving
  models directly — an unconditionally saved model once usurped a better
  one and defended itself circularly.
- Evaluation is session-split only (LOSO / aligned protocol); never random
  row splits — neighboring frames are near-duplicates.
- Python runs via `~/.venv/bin/python` (uv-managed); PyTorch for training;
  no large downloads without lazy-load and a size note.
