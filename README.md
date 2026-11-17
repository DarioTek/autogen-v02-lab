# autogen-v02

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