# autogen_sample.py
"""
AutoGen v0.2 pipeline
  1. Coder writes converter.py; reviewer reviews it until APPROVED.
  2. Coder writes test_converter.py; reviewer reviews the tests until APPROVED.
  3. In a throwaway Docker container that uses uv (no pip):
       a. ruff static check (syntax errors and undefined names)
       b. unittest suite
  4. If a check fails, its output goes back through the review loop for repair (max N times).

Requirements: pip install "pyautogen[docker]", Docker Desktop running, Ollama serving the model.
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from autogen import ConversableAgent
from autogen.coding import CodeBlock, DockerCommandLineCodeExecutor

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
script_path = Path(__file__).resolve()
clean_name = script_path.stem

# model_name = "qwen2.5-coder:3b"  # too weak for reliable repairs in testing
model_name = "qwen2.5-coder:7b"

llm_config = {
    "config_list": [
        {
            "model": model_name,
            # "base_url": "http://localhost:11434/v1",
            "base_url": "http://192.168.1.86:11434/v1",  # Raspberry Pi 5
            "api_key": "ollama",
            "price": [0, 0],  # Silences local model cost-calculation warnings
        }
    ],
    "temperature": 0.2,
    "cache_seed": None,  # AutoGen caches replies by default; disable for fresh runs
}

# Official uv image: Python 3.12 and uv preinstalled, with a shell for the check script.
DOCKER_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
CHECK_TIMEOUT_SECONDS = 300  # the first run downloads ruff and tzdata
MAX_REVIEW_TURNS = 3  # round-trips allowed in each review loop
MAX_REPAIR_ATTEMPTS = 2  # times failing checks are sent back to the coder

NO_CHANGE_NUDGE = (
    "Your last reply changed no code, but the checks still fail. "
    "You must change the code this time."
)

SPEC = """\
Write the module `converter.py` containing this function:

    def convert_epoch_timestamps(timestamps: list[int | float]) -> list[str]

Specification:
- Each input is a Unix epoch timestamp in seconds (int or float).
- Floor each timestamp to whole seconds, then return one ISO 8601 string per input in the
  America/New_York time zone, with no fractional seconds.
- Format each result with datetime.isoformat(timespec="seconds"). The UTC offset must
  contain a colon, for example "-04:00". The form "-0400" is wrong.
- Examples:
    1719835200 -> "2024-07-01T08:00:00-04:00"   (EDT, summer)
    1704110400 -> "2024-01-01T07:00:00-05:00"   (EST, winter)
    1609459200 -> "2020-12-31T19:00:00-05:00"   (2021-01-01 00:00 UTC is still 2020-12-31 in New York)
    1609459200.123 -> "2020-12-31T19:00:00-05:00"
    1612137600 -> "2021-01-31T19:00:00-05:00"
    1710053999 -> "2024-03-10T01:59:59-05:00"   (one second before DST starts)
    1710054000 -> "2024-03-10T03:00:00-04:00"   (DST starts: 2:00 EST becomes 3:00 EDT)
- Use only the standard library (zoneinfo for the time zone). Preserve input order.
- An empty list returns an empty list.
- Raise ValueError for NaN, infinity, or any value that cannot be converted to a datetime
  (convert TypeError, OverflowError and OSError into ValueError).
- Include type hints on the function and a short docstring.
"""

IMPLEMENT_TASK = (
    SPEC
    + "\nOutput the complete file in ONE python code block. The first line inside the block "
    + "must be exactly:\n# FILE: converter.py"
)

CODER_SYSTEM = (
    "You are a senior Python developer. Write clean, concise, fully type-annotated Python "
    "using only the standard library. Every code block must start with a line "
    "`# FILE: <filename>` and contain one complete file. Put each file in its own code block. "
    "Include every import the code uses. When revising, address every reviewer point and every "
    "failing check, and resend the complete updated file."
)

REVIEWER_SYSTEM = (
    "You are a strict senior code reviewer. You cannot run the code, so read it carefully. Check: "
    "(1) correctness against the specification; "
    "(2) every name used is defined or imported; "
    "(3) edge cases (empty input, NaN/infinity, wrong types, out-of-range values, DST transitions); "
    "(4) output format exactly as the specification requires, including the UTC offset format; "
    "(5) type hints on every function and parameter; "
    "(6) clear structure and docstrings, with no unnecessary code; "
    "(7) for test files, every expected value matches the specification. "
    "If the task reports failing checks, the code is NOT approved until those failures are fixed. "
    "If changes are needed, reply with a numbered list of concrete required changes and do not "
    "rewrite the code. If the code is correct and complete with no changes needed, "
    "reply with ONLY the word APPROVED."
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
# Agents
# ----------------------------------------------------------------------------
APPROVAL_RE = re.compile(r"\s*APPROVED\b")
CODE_BLOCK_RE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL)
FILE_HEADER_RE = re.compile(r"^[ \t]*#[ \t]*FILE:[ \t]*([\w.\-]+\.py)[ \t]*$", re.MULTILINE)


def is_approval(msg: dict) -> bool:
    """True if the message starts with APPROVED (so 'NOT APPROVED' does not count)."""
    return bool(APPROVAL_RE.match(msg.get("content") or ""))


coder = ConversableAgent(
    name="coder",
    system_message=CODER_SYSTEM,
    llm_config=llm_config,
    human_input_mode="NEVER",
    # The coder receives the reviewer's APPROVED, so the chat stops here.
    is_termination_msg=is_approval,
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message=REVIEWER_SYSTEM,
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def ensure_docker_running() -> None:
    """Fail fast with a clear message instead of a stack trace later."""
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit("Docker is not reachable. Start Docker Desktop and try again.") from exc


def review_loop(task: str) -> tuple[list[dict], bool]:
    """Run one review chat. Returns a copy of its transcript and whether it ended in APPROVED."""
    result = reviewer.initiate_chat(recipient=coder, message=task, max_turns=MAX_REVIEW_TURNS)
    # AutoGen clears and reuses this same message list on the next initiate_chat call,
    # so copy it now. Otherwise every saved phase shows the last conversation.
    history = [dict(msg) for msg in result.chat_history]
    approved = bool(history) and is_approval(history[-1])
    return history, approved


def split_by_header(block: str) -> dict[str, str]:
    """Split one code block into files at every '# FILE: name.py' line. Returns {} if none."""
    headers = list(FILE_HEADER_RE.finditer(block))
    files: dict[str, str] = {}
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(block)
        files[header.group(1)] = block[header.end():end]
    return files


def extract_files(history: list[dict], only: set[str]) -> dict[str, str]:
    """
    Return the newest version of each allowed file the coder produced.

    Handles three layouts: a header on the first line inside the block, a header
    above the block, and several files in one block (split at each header).
    Files outside `only` are reported and skipped.
    """
    files: dict[str, str] = {}
    for msg in history:
        # The coder's replies appear as "user" in the reviewer's history.
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
                # Fallback: the header sits on the line above the code block.
                tags = FILE_HEADER_RE.findall(preceding)
                if not tags:
                    print("Warning: a code block without a '# FILE:' header was ignored.")
                    continue
                parts = {tags[-1]: block}

            for name, body in parts.items():
                if name not in only:
                    print(f"Note: ignoring unexpected file from the coder: {name}")
                    continue
                files[name] = body.strip("\n") + "\n"
    return files


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


def build_test_task(converter_src: str) -> str:
    return (
        f"{SPEC}\n"
        f"The reviewed implementation is:\n```python\n{converter_src}\n```\n\n"
        "Write unit tests for `convert_epoch_timestamps` in a file named test_converter.py.\n"
        "Rules:\n"
        "- The first line inside the code block must be exactly `# FILE: test_converter.py`.\n"
        "- Then `import unittest` and `from converter import convert_epoch_timestamps`, "
        "plus any other import you use.\n"
        "- Use `unittest.TestCase` classes only. Do NOT add an `if __name__ == '__main__':` block.\n"
        "- Write at least six separate test methods, one per behavior: both examples, empty input, "
        "order preservation, float inputs, the DST boundary, and ValueError for NaN and infinity. "
        "Separate methods keep one failure from hiding the others.\n"
        "- Work out every expected value from the epoch in UTC, then apply the New York offset "
        "(-05:00 in winter, -04:00 in summer). Check each one against the specification examples.\n"
        "- Output only the test file, in ONE python code block."
    )


def build_repair_task(files: dict[str, str], output: str, nudge: str = "") -> str:
    current = "\n\n".join(f"File `{name}`:\n```python\n{src}\n```" for name, src in files.items())
    tail = output[-5000:]  # keep the prompt small; the ruff and test summaries are near the end
    parts = [
        SPEC,
        "The automated checks FAILED when run in an isolated Docker container "
        "(step 1 = ruff static check, step 2 = unit tests). The code is NOT approved.\n",
        f"Output (tail):\n```text\n{tail}\n```\n",
        f"Current files:\n{current}\n",
        "Fix every reported problem. Change converter.py if the implementation is wrong, or "
        "test_converter.py if an expected value contradicts the specification. "
        "Output ONLY the files you changed, each in its own python code block whose first line "
        "is `# FILE: <filename>`.",
    ]
    if nudge:
        parts.append(nudge)
    return "\n".join(parts)


def write_report(run_dir: Path, timestamp: str, phases, files, exit_code: int, output: str) -> Path:
    status = "PASSED" if exit_code == 0 else "FAILED"
    lines = [
        f"# AutoGen Run: `{clean_name}`",
        f"- **Timestamp:** {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- **Model:** `{model_name}`",
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
    for title, history, approved in phases:
        verdict = "APPROVED" if approved else "NOT APPROVED (turn limit reached)"
        lines += [f"## {title}: {verdict}", ""]
        for idx, msg in enumerate(history, start=1):
            # The reviewer opened every chat, so its own messages have role "assistant".
            speaker = "reviewer" if msg.get("role") == "assistant" else "coder"
            lines += [f"### Turn {idx}: `{speaker}`", "", (msg.get("content") or "").strip(), "", "---", ""]

    report = run_dir / f"{clean_name}_{timestamp}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> int:
    ensure_docker_running()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = script_path.parent / "output" / f"{clean_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    phases: list[tuple[str, list[dict], bool]] = []

    # 1) Implementation: the coder writes converter.py; the reviewer reviews until APPROVED.
    print("== Phase 1: implementation + review ==")
    impl_history, approved = review_loop(IMPLEMENT_TASK)
    phases.append(("Phase 1 - Implementation", impl_history, approved))
    files = extract_files(impl_history, only={"converter.py"})
    if "converter.py" not in files:
        raise RuntimeError("The coder did not return converter.py")

    # 2) Tests: the coder writes test_converter.py; the reviewer reviews the tests until APPROVED.
    print("== Phase 2: unit tests + review ==")
    test_history, approved = review_loop(build_test_task(files["converter.py"]))
    phases.append(("Phase 2 - Unit tests", test_history, approved))
    files.update(extract_files(test_history, only={"test_converter.py"}))
    if "test_converter.py" not in files:
        raise RuntimeError("The coder did not return test_converter.py")

    # 3) Checks in Docker; on failure, send the output back for repair.
    exit_code, output = 1, ""
    nudge = ""
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        write_files(run_dir, files)
        print(f"== Phase 3: checks in Docker (attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS + 1}) ==")
        exit_code, output = run_checks_in_docker(run_dir)
        print(output)
        if exit_code == 0 or attempt == MAX_REPAIR_ATTEMPTS:
            break

        print("Checks failed. Sending the output back to the coder for repair...")
        repair_history, approved = review_loop(build_repair_task(files, output, nudge))
        phases.append((f"Phase 3.{attempt + 1} - Repair", repair_history, approved))
        fixed = extract_files(repair_history, only={"converter.py", "test_converter.py"})
        if not fixed:
            print("The coder returned no usable files; stopping.")
            break
        if all(files.get(name) == src for name, src in fixed.items()):
            print("The coder returned unchanged code; the next repair will say so explicitly.")
            nudge = NO_CHANGE_NUDGE
        else:
            nudge = ""
        files.update(fixed)

    report = write_report(run_dir, timestamp, phases, files, exit_code, output)
    print(f"\nChecks: {'PASSED' if exit_code == 0 else 'FAILED'}")
    print(f"Code, check output, and report saved to: {run_dir}")
    print(f"Markdown transcript: {report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())