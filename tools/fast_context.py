"""
FastContext — delegate repository exploration to DeepSeek V4 Flash.

Runs the FastContext harness (fc_explore.py) as a subprocess with
DeepSeek V4 Flash via api.deepseek.com. Returns the <final_answer>
citation block with file paths and line ranges.

Prefer this over search_files for broad/vague queries like
"where is X implemented?" or "what files are involved in Y?"
Use search_files for exact patterns and follow-up reads.
"""

import json
import os
import subprocess
import sys
import re
import tempfile

from tools.registry import registry, tool_error

FAST_CONTEXT_SCHEMA = {
    "name": "fast_context",
    "description": (
        "Explore a codebase with a natural-language query using a FastContext "
        "subagent powered by DeepSeek V4 Flash. Returns compact file:line citations "
        "identifying relevant code locations. "
        "PREFER THIS OVER search_files for broad/vague queries like "
        "'where is X implemented?' or 'what files are involved in Y?' "
        "Use search_files for exact regex patterns and quick follow-up reads. "
        "Requires DEEPSEEK_API_KEY in environment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language exploration request. Be specific about what "
                    "you're looking for. Examples: 'Find where user authentication "
                    "is implemented', 'Locate the database migration files'"
                ),
            },
            "repo_path": {
                "type": "string",
                "description": (
                    "Absolute path to the repository root to explore. "
                    "Defaults to the current working directory."
                ),
            },
        },
        "required": ["query"],
    },
}

# Path to the FastContext harness
_HARNESS_DIR = os.path.join(
    os.path.expanduser("~"), "workspace", "fastcontext-harness"
)
_HARNESS_SCRIPT = os.path.join(_HARNESS_DIR, "fc_explore.py")


def _check_fast_context_reqs() -> bool:
    """FastContext requires DEEPSEEK_API_KEY and the harness script."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        return False
    if not os.path.isfile(_HARNESS_SCRIPT):
        return False
    return True


def _estimate_repo_turns(work_dir: str) -> int:
    """Estimate appropriate MAX_TURNS based on repo size.

    Returns 12 for medium repos (100-500 source files), 16 for large
    repos (500+), and 20 for very large repos (2000+).  Small repos
    (<100 files) use the default of 8.
    """
    try:
        # Quick file count with a ceiling and skip-list
        skip_dirs = {'node_modules', '.git', '__pycache__', '.venv', 'venv',
                     'dist', 'build', '.next', 'target', '.cache', 'coverage'}
        count = 0
        for _root, dirs, files in os.walk(work_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            count += len(files)
            if count > 3000:  # cap the walk
                break
        if count > 2000:
            return 20
        if count > 500:
            return 16
        if count > 100:
            return 12
        return 8
    except Exception:
        return 8  # safe default


def _run_fast_context(query: str, repo_path: str, task_id: str = None) -> str:
    """Run the FastContext harness and return the final answer."""
    work_dir = os.path.abspath(repo_path or ".")
    if not os.path.isdir(work_dir):
        return tool_error(f"fast_context: repo_path does not exist or is not a directory: {work_dir}")

    # Scale turn budget to repo size — larger repos need more exploration turns
    max_turns = str(_estimate_repo_turns(work_dir))

    # Collect env vars, reading API key from the process environment
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    env = os.environ.copy()
    env.update({
        "FASTCONTEXT_URL": "https://api.deepseek.com",
        "FASTCONTEXT_MODEL": "deepseek-v4-flash",
        "FASTCONTEXT_MAX_TURNS": max_turns,
        "DEEPSEEK_API_KEY": api_key,
    })

    try:
        result = subprocess.run(
            [sys.executable, _HARNESS_SCRIPT, work_dir, query],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=_HARNESS_DIR,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return tool_error("fast_context: exploration timed out after 180s")
    except Exception as e:
        return tool_error(f"fast_context: failed to run harness: {e}")

    if result.returncode != 0:
        # Check if we have partial output (non-citation mode fallback)
        if result.stdout and "<final_answer>" in result.stdout:
            pass  # We'll parse it below
        elif result.stderr:
            return tool_error(
                f"fast_context: harness failed (exit {result.returncode}). "
                f"Stderr: {result.stderr[:500]}"
            )
        else:
            return tool_error(
                f"fast_context: harness failed with exit code {result.returncode} "
                f"and no output. The repo may be too large or the query too vague."
            )

    output = (result.stdout + result.stderr).strip()
    if not output:
        return tool_error("fast_context: harness produced no output")

    # Parse <final_answer> block (model outputs this when it completes)
    match = re.search(
        r"<final_answer>(.*?)</final_answer>", output, re.DOTALL
    )
    if match:
        citations = match.group(1).strip()
        return json.dumps({
            "success": True,
            "repo": work_dir,
            "query": query,
            "citations": citations,
            "hint": "Use read_file with these paths and line ranges to read the located code.",
        })

    # Fallback: the harness also prints "FINAL ANSWER:" header
    fa_match = re.search(
        r"FINAL ANSWER:\s*\n(.*?)(?:\n={10,}|$)", output, re.DOTALL
    )
    if fa_match:
        citations = fa_match.group(1).strip()
        return json.dumps({
            "success": True,
            "repo": work_dir,
            "query": query,
            "citations": citations,
            "hint": "Use read_file with these paths and line ranges to read the located code.",
        })

    # No structured answer found — return last meaningful output
    lines = output.split("\n")
    # Try to find the last model response (before it exits)
    last_content = ""
    for line in reversed(lines):
        if line.strip() and not line.startswith("  →") and not line.startswith("    ↳"):
            last_content = line.strip()
            break

    return tool_error(
        f"fast_context: exploration completed but no citations found. "
        f"Last output: {last_content[:300]}" if last_content else
        f"fast_context: exploration failed. Output: {output[-500:]}"
    )


def _handle_fast_context(args, **kw):
    tid = kw.get("task_id") or "default"
    query = args.get("query", "")
    if not query:
        return tool_error("fast_context: query is required")
    repo_path = args.get("repo_path", ".")
    return _run_fast_context(query=query, repo_path=repo_path, task_id=tid)


# Register in the "file" toolset (alongside search_files)
registry.register(
    name="fast_context",
    toolset="file",
    schema=FAST_CONTEXT_SCHEMA,
    handler=_handle_fast_context,
    check_fn=_check_fast_context_reqs,
    emoji="🧭",
    max_result_size_chars=50_000,
    requires_env=["DEEPSEEK_API_KEY"],
)
