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

---

The core issue you're facing is that the prompts are too long and Qwen2.5-Coder-3B (despite supporting 32K context) is too weak to reliably follow complex multi-file instructions without getting confused. To optimize, you need to **drastically shorten the context** and **simplify the task structure** so the small model can focus on one thing at a time.

### 1. Shorten the System Prompts and Specs
The current `SPEC` and `REVIEWER_SYSTEM` are very verbose. A 3B model can lose track of instructions buried in long text. Cut them down to the absolute essentials.

*   **Simplify `SPEC`**: Remove redundant examples and wordy explanations. Keep only the function signature, the timezone rule, and one or two critical examples (like the DST boundary).
*   **Tighten `CODER_SYSTEM`**: Instead of a paragraph, use a terse command like: *"You are a Python coder. Output ONLY a python code block. First line: `# FILE: filename.py`. Use stdlib only. Address all reviewer points when revising."*
*   **Tighten `REVIEWER_SYSTEM`**: A 3B model struggles with long checklists. Reduce it to: *"Check code against spec. If bugs, reply with a numbered list. If correct, reply ONLY `APPROVED`."*

### 2. Reduce Conversation Turns and Context Bloat
AutoGen passes the entire chat history back to the model on each turn. For a small model, this quickly fills the context window with noise.

*   **Lower `MAX_REVIEW_TURNS`**: Change from `3` to `2` (or even `1` if you want to be aggressive). The 3B model is unlikely to fix complex issues in three back-and-forths; it will just get confused.
*   **Truncate History Manually**: In `review_loop`, you copy the history for reporting, but the agents still see the full history. You can create a custom `summary_method` or use AutoGen's transforms to compress old messages, but for a 3B model, the simplest fix is to just accept fewer turns.
*   **Check Ollama's `num_ctx`**: Ollama defaults to a small context window (often 2048 tokens) . Even if Qwen supports 32K , Ollama won't use it unless you set it. You should explicitly pass `num_ctx` in your `llm_config` or set the `OLLAMA_CONTEXT_LENGTH` environment variable on your Raspberry Pi. If the prompt exceeds the actual context window, the model's output will be garbage. Try setting `num_ctx` to `4096` or `8192` in your `llm_config`'s `config_list` item.

### 3. Simplify the Repair Loop
The current repair prompt includes the full `SPEC`, a 5000-character tail of the output, and the full current files. This is a huge amount of text for a 3B model to process.

*   **Cut the Tail**: Reduce `output[-5000:]` to `output[-1500:]`. Ruff and unittest failures are usually at the end; the model doesn't need the full log.
*   **Only Send the Broken File**: Instead of sending both `converter.py` and `test_converter.py` in the repair task, detect which one failed (from the ruff/unittest output) and send only that file. Less text = better focus for the small model.
*   **Be Brutally Direct**: Change the repair instruction to: *"Fix the error shown below. Output ONLY the corrected file in one code block with `# FILE: name.py`."*

### 4. Adjust Generation Parameters
*   **Lower Temperature**: `temperature=0.2` is okay, but for a small model, `0.0` or `0.1` forces it to be more deterministic and less likely to hallucinate code structure.
*   **Limit `max_tokens`**: In your `llm_config`, add `"max_tokens": 1500` (or similar). This prevents the model from rambling and wasting context on irrelevant text.

### 5. Code Adjustments Summary
Here are the specific changes you can make in your file:

*   **In `llm_config`**: Add `"num_ctx": 8192` and `"max_tokens": 1500`.
*   **In `SPEC`**: Remove 50% of the examples. Keep the function signature, the "floor to seconds" rule, the timezone name, and the isoformat requirement.
*   **In `REVIEWER_SYSTEM`**: Reduce to one sentence.
*   **In `build_repair_task`**: Change `output[-5000:]` to `output[-1500:]`. Remove the `SPEC` from the repair prompt (the coder already has it in context if the chat history is short, or just tell it "Follow the original spec").
*   **In `MAX_REVIEW_TURNS`**: Set to `2`.

The fundamental limitation is that a 3B model is not a great code reviewer or repair agent. By minimizing the context and making the instructions as blunt as possible, you give it the best chance of succeeding on the simpler sub-tasks (like writing a single function or fixing a syntax error), even if it can't handle the full pipeline reliably.


