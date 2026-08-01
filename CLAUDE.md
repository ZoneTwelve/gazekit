# gazekit — project rules

## Definition of done (EVERY implementation, no exceptions)

Wilson's standing rule: implementations must follow a written standard —
never ad-hoc, never "each time different". Before building anything new:

1. **Standard first** — if the feature belongs to a category with a
   standard doc (`docs/*.md`), follow it stage by stage. If it's a NEW
   category, write/extend the standard doc BEFORE the implementation.
2. **Consistent UX** — reuse the shared helpers (gates, `ui.say` voice
   prompts, verdict thresholds, checklist screens). A user should not be
   able to tell which feature was built first.
3. **Validated before "done"** — imports clean, plus an end-to-end check
   (simulated input when hardware is involved; xcodebuild for iOS).
4. **Registered** — CLI modes return a result dict so the journal records
   them; models go through deploy gates.
5. **Synced** — README updated, committed, pushed (GitHub; HF when models
   or dataset changed) in the same working session.

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
