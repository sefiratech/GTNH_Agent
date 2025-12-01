# tests/test_llm_role_boundaries.py
import sys
from subprocess import run
from pathlib import Path

def test_llm_role_boundaries():
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "audit" / "check_llm_roles.py"
    result = run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"LLM role boundary audit failed:\n{result.stdout}\n{result.stderr}"

