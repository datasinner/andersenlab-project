import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "calculate.py"
RULES = PROJECT_ROOT / "tests" / "fixtures" / "rules" / "durban"
VESSEL = PROJECT_ROOT / "tests" / "fixtures" / "vessels" / "sudestada.json"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_prints_a_report_with_formulas():
    result = _run(
        "--rules", str(RULES), "--vessel", str(VESSEL), "--port", "Durban",
        "--facts", '{"is_cargo_working": true}', "--explain",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("SUDESTADA at Durban")
    assert "147,074.38 ZAR" in result.stdout
    assert "73,118.07 + 419.12 = 73,537.19" in result.stdout
    assert "Assumptions:" in result.stdout


def test_cli_json_output():
    result = _run(
        "--rules", str(RULES), "--vessel", str(VESSEL),
        "--facts", '{"is_cargo_working": true}', "--json",
    )  # fmt: skip
    payload = json.loads(result.stdout)
    assert payload["failures"] == []
    assert payload["total"] == "509991.33"


def test_cli_reports_rules_it_cannot_evaluate():
    # Without the is_cargo_working fact, port dues can't be evaluated.
    result = _run("--rules", str(RULES), "--vessel", str(VESSEL))
    assert result.returncode == 1
    assert "port_dues: Fact 'is_cargo_working' was not resolved" in result.stderr


def test_cli_rejects_missing_files():
    result = _run("--rules", str(RULES), "--vessel", "does-not-exist.json")
    assert result.returncode == 2
    assert result.stderr.startswith("error:")
