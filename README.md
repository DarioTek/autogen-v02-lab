# autogen-v02

## Optimized pipeline for qwen2.5-coder:3b on Raspberry Pi 5 (8 GB)

`src/autogen_v02_lab/autogen_v02_coder_reviewer_A.py` is tuned for a 3B model.
Key change: **no LLM reviewer** (a 3B reviewer APPROVED wrong code in testing —
see `output/`). Correctness is enforced by deterministic gates instead:
local `compile()` + golden examples, then ruff + unittest in Docker.
Single-shot generation (`max_turns=1`), skeleton-anchored prompts,
temperature 0.0, robust code extraction (unclosed/missing fences tolerated),
and minimal repair prompts (one file + first failure, ≤1200 chars).

### On the Pi (Ollama server only — keep Docker on your laptop)

```bash
ollama pull qwen2.5-coder:3b   # Q4_K_M ≈ 1.9 GB, fits 8 GB with KV cache
OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_CTX=4096 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=10m ollama serve
```

Do not load 7B and 3B at the same time on 8 GB; `num_ctx` above 4096 wastes RAM.

### Run

```bash
uv sync   # installs autogen[openai]<1.0 + tzdata on Windows (see pyproject.toml)
uv run autogen_v02_coder_reviewer_A.py
# overrides: OLLAMA_BASE_URL=http://<pi-ip>:11434/v1 OLLAMA_MODEL=qwen2.5-coder:3b
```

Notes:
- **Do NOT** `pip install ag2[docker]` for the `docker is not installed`
  ImportError — Phase 3 uses the `docker` CLI (`docker run --rm -v …:/work`),
  not AutoGen's `DockerCommandLineCodeExecutor`, so no docker SDK is needed.
- Windows hosts have no IANA tz database: if the local golden gate reports
  `ZoneInfoNotFoundError`, that's a host issue, not a code error. `uv sync`
  installs `tzdata`, and the gate now skips (instead of asking the LLM to
  "repair" correct code) and lets Docker — which installs `tzdata` — decide.


```
# Explicitly pin the AutoGen version below 1.0
uv add "autogen[openai]<1.0"

# Run sample program connecting to raspberry pi 5 serving ollama models
uv run autogen_v02_sample.py

```

The error is still the same because the `autogen` package was not successfully installed into your `uv` virtual environment, or the package version you installed has changed the import name.

### Most likely cause: package version conflict

The `autogen` you need for your code (containing `ConversableAgent`) is the **old version (< 1.0)**. If you previously added `ag2` or `autogen` without version restrictions, `uv` may have installed the new version **1.0.0+**, which renames the top-level package to `ag2` and **removes `ConversableAgent`**, causing `from autogen import ConversableAgent` to fail directly.

### Fix steps

Execute the following commands in your project directory to explicitly pin the version below 1.0:

```bash
# First remove any potentially incorrect versions
uv remove ag2 autogen pyautogen

# Then install the old version of autogen that contains ConversableAgent
uv add "autogen[openai]<1.0"
```

After installation, confirm that `pyproject.toml` includes `autogen` with a version restriction. Then run your script again:

```bash
uv run autogen_v02_sample.py
```

### Additional notes

According to the official AG2 documentation, the classic version is published on PyPI under the name **`autogen`** (not `ag2`, which is the new version's name), and `ConversableAgent` belongs to this classic version. The official documentation also recommends installing with `pip install "autogen[openai]"`, but for `uv` environments, explicitly adding `<1.0` is the safest approach to prevent `uv` from automatically resolving to a new version that breaks your code.