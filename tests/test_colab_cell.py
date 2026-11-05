import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_cell_is_up_to_date():
    # src/ 를 고친 뒤 `python tools/build_colab_cell.py` 를 잊으면 실패
    r = subprocess.run([sys.executable, "tools/build_colab_cell.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_colab_cell_runs_in_empty_dir(tmp_path):
    cell = (ROOT / "notebooks" / "colab_one_cell.py").read_text(encoding="utf-8")
    cell = (cell.replace("CANDIDATE_SAMPLES = 12000", "CANDIDATE_SAMPLES = 2000")
                .replace("BACKTEST_WINDOW = 180", "BACKTEST_WINDOW = 30"))
    script = tmp_path / "cell.py"
    script.write_text(cell, encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "1249회 후보 2세트" in r.stdout
    assert (tmp_path / "lotto_outputs" / "run1" / "prediction_result.json").exists()
    assert (tmp_path / "lotto_results.zip").exists()
