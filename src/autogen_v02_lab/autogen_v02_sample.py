# autogen_sample.py
from datetime import datetime
from pathlib import Path
from autogen import ConversableAgent

# Get the clean script name without extension
script_path = Path(__file__).resolve()
clean_name = script_path.stem

model_name = "qwen2.5-coder:3b"
# model_name = "qwen2.5-coder:7b"

llm_config = {
    "config_list": [
        {
            "model": model_name,
            # "base_url": "http://localhost:11434/v1",
            "base_url": "http://192.168.1.86:11434/v1",  # Raspberry Pi 5
            "api_key": "ollama",
            "price": list((0, 0)),  # Silences local model cost-calculation warnings
        }
    ],
    "temperature": 1.0,
}

coder = ConversableAgent(
    name="coder",
    system_message="You are a senior Python developer. Write clean, concise Python code.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message=(
        "You are a code reviewer. Check for edge cases and type hints. "
        "If no changes are needed, reply ONLY with 'APPROVED'."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "APPROVED" in (msg.get("content") or ""),
)

chat_result = reviewer.initiate_chat(
    recipient=coder,
    message="Write a Python function that parses a list of epoch timestamps and converts them to America/New_York ISO strings.",
    max_turns=4,
)

# Create the output folder next to the script if it doesn't exist
output_dir = script_path.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Build the timestamped filename: autogen_sample_YYYYMMDD_HHMMSS.md
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"{clean_name}_{time_stamp}.md"

# Format and save the conversation history as Markdown
md_lines = [
    f"# AutoGen Run: `{clean_name}`",
    f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"- **Model:** `{model_name}`",
    "",
    "---",
    "",
]

for idx, msg in enumerate(chat_result.chat_history, start=1):
    speaker = msg.get("name") or msg.get("role", "unknown")
    content = (msg.get("content") or "").strip()
    md_lines.extend([
        f"## Turn {idx}: `{speaker}`",
        "",
        content,
        "",
        "---",
        "",
    ])

output_file.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\nSaved run output to: {output_file}")
