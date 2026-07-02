# Google Colab / Jupyter 실행용: 프로젝트 루트 자동 탐색 + src import 오류 방지
!pip -q install pandas numpy matplotlib requests python-dotenv lxml openpyxl streamlit

from pathlib import Path
import zipfile, os, sys, json, shutil

PROJECT_NAME = "korea_lotto_agent_harness_v6"
ZIP_NAMES = [
    "/content/korea_lotto_agent_harness_v6.zip",
    "/content/korea_lotto_agent_harness_v6_1_colab_fix.zip",
    str(Path.cwd() / "korea_lotto_agent_harness_v6.zip"),
    str(Path.cwd() / "korea_lotto_agent_harness_v6_1_colab_fix.zip"),
]

# 1) ZIP이 있으면 압축 해제
zip_found = None
for zp in ZIP_NAMES:
    if Path(zp).exists():
        zip_found = Path(zp)
        break

# Colab에서 ZIP을 아직 업로드하지 않았다면 업로드 창 표시
if zip_found is None and Path('/content').exists():
    try:
        from google.colab import files
        print("ZIP 파일을 업로드하세요: korea_lotto_agent_harness_v6.zip")
        uploaded = files.upload()
        for name in uploaded.keys():
            if name.endswith('.zip'):
                zip_found = Path('/content') / name
                break
    except Exception as e:
        print("Colab files.upload()를 사용할 수 없습니다:", e)

if zip_found is not None:
    print("압축 해제:", zip_found)
    with zipfile.ZipFile(zip_found, "r") as z:
        z.extractall("/content" if Path('/content').exists() else Path.cwd())
else:
    print("ZIP 파일을 찾지 못했습니다. 이미 압축 해제된 폴더를 탐색합니다.")

# 2) src/config.py가 있는 실제 프로젝트 루트 자동 탐색
def find_project_root():
    roots = []
    for base in [Path.cwd(), Path('/content'), Path('/mnt/data')]:
        if base.exists():
            roots.append(base)
    seen = set()
    for base in roots:
        for p in base.rglob('src/config.py'):
            root = p.parent.parent
            if root not in seen:
                seen.add(root)
                # app.py 또는 cli.py가 있는 프로젝트를 우선 선택
                if (root / 'app.py').exists() or (root / 'cli.py').exists():
                    return root
    return None

PROJECT = find_project_root()
if PROJECT is None:
    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. ZIP을 /content에 업로드했는지, "
        "압축 해제 후 src/config.py가 존재하는지 확인하세요."
    )

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print("프로젝트 루트:", PROJECT)
print("현재 작업 폴더:", Path.cwd())
print("src/config.py 존재:", (PROJECT / 'src' / 'config.py').exists())

# 3) 기본 실행
from src.config import AppConfig
from src.harness.base import HarnessContext
from src.harness.orchestrator import AgentHarness

cfg = AppConfig.from_env(
    target_round=1226,
    n_sets=2,
    candidate_samples=12000,
    backtest_window=180,
    output_dir="/content/lotto_outputs" if Path('/content').exists() else str(PROJECT / 'outputs_colab')
)
ctx = HarnessContext(
    cfg=cfg,
    csv_path=str(PROJECT / "data" / "korealotto_1224V_CSV2.csv"),
    xls_path=str(PROJECT / "data" / "LottoSpecial_공나온순서_1225.xls")
)
ctx = AgentHarness(include_llm=False).run(ctx)

result = ctx.artifacts["pipeline"]
print("\n1226회차 후보 2세트")
for i, s in enumerate(result["final_sets"], 1):
    print(f"Set {i}: {s}")

print("\n에이전트 실행 결과")
for ar in ctx.agent_results:
    print(ar.name, ar.status, ar.message)

print("\n저장 경로:", cfg.output_dir)
print([str(p) for p in Path(cfg.output_dir).glob("*")])