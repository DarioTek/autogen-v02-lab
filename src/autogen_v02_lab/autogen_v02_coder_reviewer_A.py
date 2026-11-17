# autogen_v02_coder_reviewer_A.py
"""
AutoGen v0.2 pipeline — OPTIMIZED for qwen2.5-coder:3b on a Raspberry Pi 5 (8 GB RAM).

Why this version differs from the 7B original (each change targets a known 3B failure mode):

  1. NO LLM REVIEWER. The 3B reviewer rubber-stamps wrong code as APPROVED
     (observed: pytz + wrong function name + strftime('%z') approved). A weak
     model cannot review itself. Review is replaced by deterministic Python
     gates: compile() + golden examples + ruff + unittest in Docker.
     This also halves the number of LLM calls on the Pi.

  2. SINGLE-SHOT CODER + DETERMINISTIC REPAIR LOOP (max_turns=1 per call).
     Multi-turn review chats blow past the 3B context window and the model
     forgets the spec. Each repair prompt is now minimal: only the failing
     file + the first failure (<= ~1200 chars), NOT spec + both files + 5k
     chars of output. Context stays well under 4k tokens.

  3. SKELETON ANCHORING. The coder gets a fill-in stub with exact imports,
     signature, and docstring instead of an abstract spec alone. Small
     models copy templates far more reliably than they follow prose specs.

  4. SHORT, NEGATIVE-CONSTRAINT PROMPTS. Long system prompts are ignored by
     3B. The prompt is ~15 lines and explicitly bans the observed failures:
     pytz, strftime, pip install, prose outside the code block.

  5. ROBUST EXTRACTION. 3B often omits the closing fence, the '# FILE:'
     header, or wraps code in prose. extract_code() handles all three
     instead of discarding the reply and wasting a Docker run.

  6. LOCAL PRE-CHECKS BEFORE DOCKER. compile() + golden cases run in-process
     in <1s and give precise repair feedback. Docker is kept only as the
     final isolation gate (ruff + unittest), so most failures never pay the
     container startup cost.

  7. PI-FRIENDLY OLLAMA SETTINGS. temperature=0.0, capped max_tokens,
     explicit timeout. Server-side (see README / header below):
       - Pull the quantized model:  ollama pull qwen2.5-coder:3b
         (Q4_K_M ≈ 1.9 GB — fits comfortably in 8 GB with KV cache)
       - Serve with:  OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_CTX=4096 \
                      OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=10m ollama serve
       - num_ctx=4096 is enough once prompts are short; 8192+ wastes Pi RAM
         on KV cache. Never load 7B and 3B at the same time on 8 GB.
       - Docker stays on the laptop, Ollama stays on the Pi. Do not run
         Docker on the Pi for this pipeline.

Requirements: pip install "pyautogen[docker]", Docker Desktop running,
Ollama serving qwen2.5-coder:3b on the Pi.
Env overrides: OLLAMA_BASE_URL (default http://192.168.1.86:11434/v1),
               OLLAMA_MODEL (default qwen2.5-coder:3b).
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from autogen import ConversableAgent, UserProxyAgent
from autogen.coding import CodeBlock, DockerCommandLineCodeExecutor

# ----------------------------------------------------------------------------
# Configuration (3B-tuned)
# ----------------------------------------------------------------------------
script_path = Path(__file__).resolve()
clean_name = script_path.stem

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.86:11434/v1")

# 3B needs deterministic output. 0.2 (7B setting) is enough to drift off-spec.
TEMPERATURE = 0.0
# Cap output so the model cannot ramble/truncate the code block. Converter
# needs ~300 tokens; tests ~600. Larger caps waste Pi time and invite loops.
MAX_TOKENS_CODER = 1024
MAX_TOKENS_TESTS = 1024
LLM_TIMEOUT_SECONDS = 300  # Pi 5 generation is slow; do not time out early.

llm_config = {
    "config_list": [
        {
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "api_key": "ollama",
            "price": [0, 0],  # Silences local model cost-calculation warnings
        }
    ],
    "temperature": TEMPERATURE,
    "timeout": LLM_TIMEOUT_SECONDS,
    "cache_seed": None,  # AutoGen caches replies by default; disable for fresh runs
}

# Official uv image: Python 3.12 and uv preinstalled, with a shell for the check script.
DOCKER_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
CHECK_TIMEOUT_SECONDS = 300  # the first run downloads ruff and tzdata
MAX_REPAIRS = 3  # single-shot repairs per phase (cheap now that reviewer is gone)

# ----------------------------------------------------------------------------
# Task: condensed spec + skeleton (3B follows templates, not prose)
# ----------------------------------------------------------------------------
# Golden cases are the contract. They are checked deterministically in Python
# (locally and in Docker) — the model never has to invent expected values.
GOLDEN_CASES: list[tuple[list, list[str]]] = [
    ([1719835200], ["2024-07-01T08:00:00-04:00"]),          # EDT, summer
    ([1704110400], ["2024-01-01T07:00:00-05:00"]),          # EST, winter
    ([1609459200], ["2020-12-31T19:00:00-05:00"]),          # UTC midnight still 2020 in NY
    ([1609459200.123], ["2020-12-31T19:00:00-05:00"]),      # floor fractional seconds
    ([1612137600], ["2021-01-31T19:00:00-05:00"]),
    ([1710053999], ["2024-03-10T01:59:59-05:00"]),          # 1s before DST
    ([1710054000], ["2024-03-10T03:00:00-04:00"]),          # DST: 2:00 EST -> 3:00 EDT
    ([], []),                                                # empty in, empty out
    ([1704110400, 1719835200], ["2024-01-01T07:00:00-05:00", "2024-07-01T08:00:00-04:00"]),  # order
]

SKELETON = """\
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def convert_epoch_timestamps(timestamps: list[int | float]) -> list[str]:
    \"\"\"Convert epoch seconds to America/New_York ISO 8601 strings.\"\"\"
    # TODO: implement. Floor to whole seconds, use datetime.fromtimestamp
    # with tz=NY, then .isoformat(timespec="seconds").
    # Raise ValueError for NaN, infinity, or unconvertible values.
    raise NotImplementedError
"""

IMPLEMENT_TASK = f"""\
Write converter.py. Fill in the TODO in this skeleton. Do not change the
imports, the function name, or the signature.

Skeleton:
```python
{SKELETON}```

Rules:
- Output ONLY one python code block. No prose before or after it.
- First line inside the block must be exactly: # FILE: converter.py
- Use only: datetime, zoneinfo, math. Do NOT use pytz. Do NOT use strftime.
- Format with .isoformat(timespec="seconds") so the offset has a colon (-04:00).
- Floor each timestamp to whole seconds. Preserve input order.
- Empty list returns empty list.
- Raise ValueError for NaN, infinity, or any unconvertible value
  (wrap TypeError, OverflowError, OSError into ValueError).

Required results (your code must produce exactly these):
{chr(10).join(f"- {inp} -> {exp}" for inp, exp in GOLDEN_CASES)}
"""

TEST_TASK_TEMPLATE = """\
Write test_converter.py for convert_epoch_timestamps. Copy the expected
values below VERBATIM — do not recompute them.

Skeleton:
```python
# FILE: test_converter.py
import math
import unittest

from converter import convert_epoch_timestamps


class TestConvertEpochTimestamps(unittest.TestCase):
    # TODO: add test methods below. Do NOT add an if __name__ block.
    pass
```

Rules:
- Output ONLY one python code block. No prose before or after it.
- First line inside the block must be exactly: # FILE: test_converter.py
- Use unittest.TestCase only. One behavior per test method.
- Required tests (copy expected values exactly):
  1. test_summer_edt: [1719835200] -> ["2024-07-01T08:00:00-04:00"]
  2. test_winter_est: [1704110400] -> ["2024-01-01T07:00:00-05:00"]
  3. test_empty: [] -> []
  4. test_float_floor: [1609459200.123] -> ["2020-12-31T19:00:00-05:00"]
  5. test_dst_boundary: [1710053999, 1710054000] ->
     ["2024-03-10T01:59:59-05:00", "2024-03-10T03:00:00-04:00"]
  6. test_nan_raises: assertRaises(ValueError) for [math.nan]
  7. test_inf_raises: assertRaises(ValueError) for [math.inf]
"""

# 3B ignores long system prompts. Keep it to what the model must not forget.
CODER_SYSTEM = (
    "You write Python. Output ONLY code blocks. No prose, no explanations, "
    "no pip install commands. Every code block starts with `# FILE: <name>.py` "
    "and contains one complete file. Use only the standard library."
)

# The script run inside the container. Dependencies come from uv, not pip.
# `uv run --with` installs into a temporary environment, so nothing is installed globally.
CHECK_SCRIPT = """\
export PYTHONDONTWRITEBYTECODE=1
echo "== Step 1/2: static check (ruff: syntax errors and undefined names) =="
uv run --quiet --with ruff ruff check --no-cache --select E9,F63,F7,F82 converter.py test_converter.py
static_exit=$?
echo "== Step 2/2: unit tests =="
uv run --quiet --with tzdata python -m unittest -v test_converter
tests_exit=$?
echo "static_check_exit=$static_exit unit_tests_exit=$tests_exit"
[ "$static_exit" -eq 0 ] && [ "$tests_exit" -eq 0 ]
"""

# ----------------------------------------------------------------------------
# Agents (coder only — no LLM reviewer for 3B)
# ----------------------------------------------------------------------------
CODE_BLOCK_RE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL)
UNCLOSED_BLOCK_RE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*)", re.DOTALL)
FILE_HEADER_RE = re.compile(r"^[ \t]*#[ \t]*FILE:[ \t]*([\w.\-]+\.py)[ \t]*$", re.MULTILINE)


def make_coder(max_tokens: int) -> ConversableAgent:
    """Fresh coder per call so max_tokens can differ (converter vs tests)."""
    cfg = dict(llm_config)
    cfg["max_tokens"] = max_tokens
    return ConversableAgent(
        name="coder",
        system_message=CODER_SYSTEM,
        llm_config=cfg,
        human_input_mode="NEVER",
        code_execution_config=False,
    )


# Initiator only: sends the task, never generates. max_turns=1 = single shot.
task_runner = UserProxyAgent(
    name="task_runner",
    human_input_mode="NEVER",
    code_execution_config=False,
)


def single_shot(task: str, max_tokens: int) -> list[dict]:
    """One coder generation, no review loop. Returns a copy of the transcript."""
    coder = make_coder(max_tokens)
    result = task_runner.initiate_chat(recipient=coder, message=task, max_turns=1)
    return [dict(msg) for msg in result.chat_history]


# ----------------------------------------------------------------------------
# Robust extraction (3B often drops fences/headers)
# ----------------------------------------------------------------------------
def split_by_header(block: str) -> dict[str, str]:
    """Split one code block into files at every '# FILE: name.py' line."""
    headers = list(FILE_HEADER_RE.finditer(block))
    files: dict[str, str] = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(block)
        files[header.group(1)] = block[header.end():end]
    return files


def _strip_prose_fallback(text: str) -> str | None:
    """Last resort: code without fences. Take from the first code-like line."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if re.match(r"\s*(import |from |def |class |# FILE:)", ln)),
        None,
    )
    if start is None:
        return None
    body = "\n".join(lines[start:]).strip("\n")
    # Drop trailing prose: stop at a line that is clearly not code and
    # follows a blank line (common 3B pattern: code, blank, "Explanation:").
    out: list[str] = []
    for ln in body.splitlines():
        if re.match(r"^(Explanation|Note|Hope|Let me|Here)", ln.strip()):
            break
        out.append(ln)
    code = "\n".join(out).strip("\n") + "\n"
    return code if "def " in code or "import " in code else None


def extract_code(history: list[dict], filename: str) -> str | None:
    """
    Return the newest version of `filename` from the coder's replies.

    Handles: header inside block, header above block, several files in one
    block, unclosed fence, and code with no fence at all.
    """
    found: str | None = None
    for msg in history:
        # The coder's replies appear as "user" in the task_runner history.
        if msg.get("role") != "user":
            continue
        text = msg.get("content") or ""
        cursor = 0
        for match in CODE_BLOCK_RE.finditer(text):
            preceding = text[cursor:match.start()]
            cursor = match.end()
            block = match.group(1)
            parts = split_by_header(block)
            if not parts:
                tags = FILE_HEADER_RE.findall(preceding)
                if tags and tags[-1] == filename:
                    found = block.strip("\n") + "\n"
                elif not tags and filename == "converter.py":
                    # Single unlabeled block: assume it is the converter.
                    # (Only safe for phase 1; tests are gated separately.)
                    found = block.strip("\n") + "\n"
                continue
            if filename in parts:
                found = parts[filename].strip("\n") + "\n"
        if found is not None:
            continue
        # No closed block matched: try unclosed fence, then raw fallback.
        unclosed = UNCLOSED_BLOCK_RE.search(text)
        if unclosed:
            block = unclosed.group(1)
            parts = split_by_header(block)
            if filename in parts:
                found = parts[filename].strip("\n") + "\n"
                continue
            if not parts:
                found = block.strip("\n") + "\n"
                continue
        fallback = _strip_prose_fallback(text)
        if fallback is not None:
            found = fallback
    return found


# ----------------------------------------------------------------------------
# Deterministic local gates (replace the LLM reviewer)
# ----------------------------------------------------------------------------
def check_syntax(src: str, filename: str) -> str | None:
    """Return a syntax-error string, or None if the file compiles."""
    try:
        compile(src, filename, "exec")
    except SyntaxError as exc:
        return f"SyntaxError in {filename}: {exc.msg} (line {exc.lineno})"
    return None


def run_golden(src: str) -> str | None:
    """Run GOLDEN_CASES against the generated converter. None = all pass."""
    namespace: dict = {}
    try:
        exec(compile(src, "converter.py", "exec"), namespace)  # noqa: S102 - lab code
    except Exception as exc:  # noqa: BLE001 - report any import-time failure
        return f"converter.py failed to import: {type(exc).__name__}: {exc}"
    fn = namespace.get("convert_epoch_timestamps")
    if not callable(fn):
        return "converter.py does not define convert_epoch_timestamps()"
    for inp, expected in GOLDEN_CASES:
        try:
            got = fn(list(inp))
        except Exception as exc:  # noqa: BLE001
            return f"convert_epoch_timestamps({inp!r}) raised {type(exc).__name__}: {exc}"
        if got != expected:
            return f"convert_epoch_timestamps({inp!r}) -> {got!r}, expected {expected!r}"
    # Error cases: NaN / inf / wrong type must raise ValueError.
    for bad in ([math.nan], [math.inf], ["bad"], [None]):
        try:
            fn(list(bad))
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            return f"convert_epoch_timestamps({bad!r}) raised {type(exc).__name__}, expected ValueError"
        return f"convert_epoch_timestamps({bad!r}) did not raise ValueError"
    return None


def validate_converter(src: str) -> str | None:
    """Syntax + golden gate for converter.py. None = pass."""
    err = check_syntax(src, "converter.py")
    if err:
        return err
    return run_golden(src)


def validate_tests(src: str) -> str | None:
    """Syntax + structure gate for test_converter.py. None = pass."""
    err = check_syntax(src, "test_converter.py")
    if err:
        return err
    if "from converter import convert_epoch_timestamps" not in src:
        return "test_converter.py must import convert_epoch_timestamps from converter"
    if src.count("def test_") < 6:
        return "test_converter.py must define at least 6 test methods (def test_...)"
    return None


# ----------------------------------------------------------------------------
# Prompts (minimal repair prompts keep 3B in-context)
# ----------------------------------------------------------------------------
def build_repair_task(filename: str, current_src: str, failure: str) -> str:
    """One file + first failure only. Never resend spec + both files + 5k output."""
    failure = failure.strip()[-1200:]
    return f"""\
Fix {filename}. It failed this check:
```text
{failure}
```

Current {filename}:
```python
{current_src}
```

Rules:
- Output ONLY the fixed {filename} in one python code block.
- First line inside the block must be exactly: # FILE: {filename}
- Keep the function name and signature unchanged.
- Use zoneinfo and .isoformat(timespec="seconds"). Do NOT use pytz or strftime.
"""


# ----------------------------------------------------------------------------
# Docker + report helpers
# ----------------------------------------------------------------------------
def ensure_docker_running() -> None:
    """Fail fast with a clear message instead of a stack trace later."""
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("Docker is not reachable. Start Docker Desktop and try again.") from exc


def check_ollama() -> None:
    """Warn early if the Pi is unreachable, instead of failing mid-pipeline."""
    import urllib.request

    models_url = BASE_URL.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Ollama not reachable at {BASE_URL} ({exc}).")
        print("  On the Pi: ollama list  (must show qwen2.5-coder:3b)")
        print("  Serve with: OLLAMA_HOST=0.0.0.0 OLLAMA_NUM_CTX=4096 ollama serve")
        return
    if MODEL_NAME.split(":")[0] not in body:
        print(f"Warning: model '{MODEL_NAME}' not listed by Ollama at {BASE_URL}.")
        print("  On the Pi: ollama pull qwen2.5-coder:3b")


def write_files(run_dir: Path, files: dict[str, str]) -> None:
    for name, src in files.items():
        (run_dir / name).write_text(src, encoding="utf-8")


def run_checks_in_docker(run_dir: Path) -> tuple[int, str]:
    """Run the static check and the unit tests inside a fresh, isolated container."""
    with DockerCommandLineCodeExecutor(
        image=DOCKER_IMAGE,
        timeout=CHECK_TIMEOUT_SECONDS,
        work_dir=run_dir,  # bind-mounted into the container as the working directory
    ) as executor:
        result = executor.execute_code_blocks([CodeBlock(code=CHECK_SCRIPT, language="sh")])
    return result.exit_code, result.output


def write_report(
    run_dir: Path, timestamp: str, phases: list[tuple[str, list[dict]]],
    files: dict[str, str], exit_code: int, output: str,
) -> Path:
    status = "PASSED" if exit_code == 0 else "FAILED"
    lines = [
        f"# AutoGen Run: `{clean_name}` (3B-optimized, no LLM reviewer)",
        f"- **Timestamp:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- **Model:** `{MODEL_NAME}` at `{BASE_URL}` (temperature={TEMPERATURE})",
        f"- **Docker image:** `{DOCKER_IMAGE}`",
        f"- **Static check + unit tests:** **{status}** (exit code {exit_code})",
        "",
        "## Final check output",
        "```text",
        output.strip(),
        "```",
        "",
    ]
    for name, src in files.items():
        lines += [f"## Final `{name}`", "```python", src.strip(), "```", ""]
    for title, history in phases:
        lines += [f"## {title}", ""]
        for idx, msg in enumerate(history, start=1):
            speaker = "coder" if msg.get("role") == "user" else msg.get("role", "unknown")
            lines += [f"### Turn {idx}: `{speaker}`", "", (msg.get("content") or "").strip(), "", "---", ""]
    report = run_dir / f"{clean_name}_{timestamp}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# ----------------------------------------------------------------------------
# Phase runners: single-shot generate -> deterministic gate -> minimal repair
# ----------------------------------------------------------------------------
def generate_with_repairs(
    task: str,
    filename: str,
    validator,
    max_tokens: int,
    phases: list[tuple[str, list[dict]]],
    label: str,
) -> str:
    """Generate `filename` once, then repair against the validator (no reviewer)."""
    history = single_shot(task, max_tokens)
    phases.append((f"{label} - initial", history))
    src = extract_code(history, filename)
    if src is None:
        raise RuntimeError(f"The coder did not return {filename}")

    for attempt in range(MAX_REPAIRS):
        failure = validator(src)
        if failure is None:
            print(f"{filename}: local gate PASSED")
            return src
        print(f"{filename}: local gate FAILED ({failure})")
        if attempt == MAX_REPAIRS - 1:
            break
        print(f"  Repair {attempt + 1}/{MAX_REPAIRS - 1}...")
        history = single_shot(build_repair_task(filename, src, failure), max_tokens)
        phases.append((f"{label} - repair {attempt + 1}", history))
        fixed = extract_code(history, filename)
        if fixed is None:
            print(f"  Coder returned no usable {filename}; keeping previous version.")
            continue
        if fixed == src:
            print("  Coder returned unchanged code; next repair will repeat the failure.")
        src = fixed

    # Return the best effort; Docker is the final arbiter and will report honestly.
    return src


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> int:
    ensure_docker_running()
    check_ollama()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = script_path.parent / "output" / f"{clean_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    phases: list[tuple[str, list[dict]]] = []

    # 1) Implementation: single-shot + golden gate (no LLM reviewer).
    print("== Phase 1: converter.py (single-shot + golden gate) ==")
    converter_src = generate_with_repairs(
        IMPLEMENT_TASK, "converter.py", validate_converter,
        MAX_TOKENS_CODER, phases, "Phase 1",
    )
    files: dict[str, str] = {"converter.py": converter_src}

    # 2) Tests: single-shot + structure gate. Expected values are pinned in
    #    the prompt (copy verbatim), so the model never invents them.
    print("== Phase 2: test_converter.py (single-shot + structure gate) ==")
    test_src = generate_with_repairs(
        TEST_TASK_TEMPLATE, "test_converter.py", validate_tests,
        MAX_TOKENS_TESTS, phases, "Phase 2",
    )
    files["test_converter.py"] = test_src

    # 3) Checks in Docker; on failure, repair the failing file with a minimal
    #    prompt (first failure only) and re-run. No review loop.
    exit_code, output = 1, ""
    for attempt in range(MAX_REPAIRS + 1):
        write_files(run_dir, files)
        print(f"== Phase 3: checks in Docker (attempt {attempt + 1}/{MAX_REPAIRS + 1}) ==")
        exit_code, output = run_checks_in_docker(run_dir)
        print(output)
        if exit_code == 0 or attempt == MAX_REPAIRS:
            break

        # Decide which file to repair from the failure text (ruff names the
        # file; unittest names the test). Default to converter.py.
        target = "test_converter.py" if "test_converter" in output and "converter.py:" not in output else "converter.py"
        if "converter.py" in output and "test_converter" in output:
            # Both mentioned: prefer the file ruff flagged first.
            target = "converter.py" if output.find("converter.py") < output.find("test_converter") else "test_converter.py"
        print(f"Checks failed. Repairing {target} with a minimal prompt...")
        history = single_shot(build_repair_task(target, files[target], output), MAX_TOKENS_CODER)
        phases.append((f"Phase 3.{attempt + 1} - Docker repair ({target})", history))
        fixed = extract_code(history, target)
        if fixed is None:
            print(f"The coder returned no usable {target}; stopping.")
            break
        if fixed == files[target]:
            print("The coder returned unchanged code; stopping to avoid a loop.")
            break
        files[target] = fixed

    report = write_report(run_dir, timestamp, phases, files, exit_code, output)
    print(f"\nChecks: {'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"Code, check output, and report saved to: {run_dir}")
    print(f"Markdown transcript: {report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
