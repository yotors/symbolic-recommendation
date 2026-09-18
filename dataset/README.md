# Local datasets

Dataset and generated replay files are intentionally excluded from Git.

Expected files for the retained MIND demonstration:

- `MIND_small_x1.zip`
- `MIND_small_x1.text-embeddings.npz`
- `llm-workspace-canonical-v3.json.gz`
- `.mind-replay-46d0a7c99fe3306f733d.json.gz` (optional loader cache)

Run `python -m recommendation.cli.fetch_mind` to fetch MIND-small, then use the
projection commands documented in the project architecture when rebuilding the
derived workspace.
