# @title 🎯 Korea Lotto Agent Harness v7 — Colab 단일 셀 실행
# 이 셀 하나만 Colab 에 붙여넣고 실행(▶)하세요. 프로그램 전체와 기본 데이터(1~1224회)가 셀 안에 들어 있습니다.
# 공정 추첨에서 모든 6개 조합의 1등 확률은 1/8,145,060 으로 같습니다. 이 도구는 통계 연구용입니다.

# ── 실행 설정 (Colab 오른쪽 폼에서도 바꿀 수 있습니다) ─────────────────────────
TARGET_ROUND = 1249  # @param {type:"integer"}
N_SETS = 2  # @param {type:"integer"}
CANDIDATE_SAMPLES = 12000  # @param {type:"integer"}
BACKTEST_WINDOW = 180  # @param {type:"integer"}
MAX_OVERLAP = 2  # @param {type:"integer"}
OPTIMIZE = True  # @param {type:"boolean"}
UPLOAD_DATA = True  # @param {type:"boolean"}
# ↑ True: 실행 중 업로드 창이 뜹니다. 최신 회차 CSV 또는 공나온순서 XLS/HTML 을 올리면 기본 데이터와 합쳐집니다. 취소하면 기본 데이터만 사용.

# ── AI 리뷰 (분석가 → 비평가 → 최적화 자문) ───────────────────────────────────
LLM_PROVIDER = "none"  # @param ["none", "hf_api", "claude_cli"]
HF_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"  # @param {type:"string"}
CLAUDE_MODEL = "sonnet"  # @param ["sonnet", "opus", "haiku", "fable", "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
APPLY_AI_SUGGESTIONS = False  # @param {type:"boolean"}
# ↑ True: AI 자문 제안(허용 범위로 제한)을 적용해 한 번 더 실행하고 결과를 비교합니다.
# 토큰은 Colab 왼쪽 🔑(보안 비밀)에 저장하고 '노트북 액세스'를 켜세요:
#   hf_api     → HF_TOKEN
#   claude_cli → ANTHROPIC_API_KEY (Claude Code CLI 가 없으면 npm 으로 자동 설치)

# ── 결과 ─────────────────────────────────────────────────────────────────────
DOWNLOAD_RESULTS = True  # @param {type:"boolean"}

# ═════════════════════════ 이하 수정 불필요 ═════════════════════════
import base64, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

_BUNDLE_VERSION = "__BUNDLE_VERSION__"
_BUNDLE = "__BUNDLE__"

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

ROOT = Path.cwd() / "korea_lotto_harness_v7"
OUT_DIR = Path.cwd() / "lotto_outputs"


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


NOTEBOOK = _in_notebook()


def _show(df):
    if NOTEBOOK:
        from IPython.display import display
        display(df)
    else:
        print(df.to_string())


def _markdown(text):
    if NOTEBOOK:
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)


def _secret(name):
    if IN_COLAB:
        try:
            from google.colab import userdata
            v = userdata.get(name)
            if v:
                return v
        except Exception:
            pass
    return os.environ.get(name, "")


# 1) 프로그램 풀기 (실행할 때마다 새로 풀어서 셀 내용과 항상 일치)
shutil.rmtree(ROOT / "src", ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_BUNDLE))) as z:
    z.extractall(ROOT)
for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[name]  # 셀 재실행 시 이전 코드 캐시 제거
sys.path.insert(0, str(ROOT))
print(f"✅ 프로그램 준비 완료 (v7 bundle {_BUNDLE_VERSION}) → {ROOT}")

from src.config import AppConfig  # noqa: E402
from src.harness.base import HarnessContext  # noqa: E402
from src.harness.orchestrator import AgentHarness  # noqa: E402
from src.harness.review import apply_suggestions  # noqa: E402

# 2) 데이터
csv_path = str(ROOT / "data" / "korealotto_1224V_CSV2.csv")
xls_path = None
if UPLOAD_DATA and IN_COLAB:
    from google.colab import files
    print("📂 추가 데이터(CSV 또는 공나온순서 XLS/HTML)를 올리세요. 없으면 '취소'를 누르세요.")
    try:
        uploaded = files.upload()
    except Exception as e:
        print("업로드 건너뜀:", e)
        uploaded = {}
    up_dir = Path.cwd() / "lotto_uploads"
    up_dir.mkdir(exist_ok=True)
    for fname, data in uploaded.items():
        p = up_dir / Path(fname).name
        p.write_bytes(data)
        if p.suffix.lower() == ".csv":
            csv_path = str(p)
        elif p.suffix.lower() in (".xls", ".html", ".htm"):
            xls_path = str(p)
        print("  ·", p.name)
    if not csv_path.startswith(str(ROOT)):
        print("  업로드한 CSV(전체 회차 형식)를 기본 CSV 대신 사용합니다.")
    if xls_path:
        print("  업로드한 XLS/HTML 회차를 CSV 와 합칩니다 (같은 회차는 XLS 값 우선).")

# 3) LLM 준비
provider = LLM_PROVIDER
if provider == "hf_api":
    token = _secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        print("⚠️ HF_TOKEN 이 없습니다. 🔑 보안 비밀에 HF_TOKEN 을 추가하고 노트북 액세스를 켜세요. → AI 리뷰 생략")
        provider = "none"
elif provider == "claude_cli":
    key = _secret("ANTHROPIC_API_KEY")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    if not shutil.which("claude"):
        print("⏳ Claude Code CLI 설치 중 (npm install -g @anthropic-ai/claude-code)...")
        r = subprocess.run(["npm", "install", "-g", "@anthropic-ai/claude-code"], capture_output=True, text=True)
        if r.returncode != 0:
            print("⚠️ 설치 실패:", (r.stderr or r.stdout)[-500:])
    if not shutil.which("claude"):
        print("⚠️ Claude Code CLI 를 사용할 수 없습니다 → AI 리뷰 생략")
        provider = "none"
    elif not key:
        print("⚠️ ANTHROPIC_API_KEY 가 없습니다. 🔑 보안 비밀에 추가하세요 → AI 리뷰 생략")
        provider = "none"


def build_cfg(output_dir):
    return AppConfig.from_env(
        target_round=int(TARGET_ROUND), n_sets=int(N_SETS), candidate_samples=int(CANDIDATE_SAMPLES),
        backtest_window=int(BACKTEST_WINDOW), max_overlap=int(MAX_OVERLAP), optimize=bool(OPTIMIZE),
        output_dir=str(output_dir), llm_provider=provider, hf_router_model=HF_MODEL, claude_model=CLAUDE_MODEL,
    )


def run(cfg, title):
    print(f"\n━━━━━━━━ {title} ━━━━━━━━")
    ctx = HarnessContext(cfg=cfg, csv_path=csv_path, xls_path=xls_path, on_progress=lambda m: print("  ·", m))
    return AgentHarness(include_llm=provider != "none").run(ctx)


def report(ctx):
    res = ctx.artifacts.get("pipeline", {})
    if not res:
        for a in ctx.agent_results:
            print(f"- {a.name}: {a.status} | {a.message}")
        return
    opt, audit = res["optimization"], ctx.artifacts.get("audit", {})
    ho = opt.get("holdout") or {}
    _markdown(f"## 🎯 {ctx.cfg.target_round}회 후보 {ctx.cfg.n_sets}세트\n\n"
              + "\n".join(f"- **Set {i}:** {' · '.join(map(str, s))}" for i, s in enumerate(res["final_sets"], 1)))
    if ho.get("holdout_folds"):
        b, d = ho["baseline_equal_weights"], ho["ensemble_minus_equal_edge_ci95"]
        _markdown(
            f"### 검증 (튜닝에 쓰지 않은 {ho['holdout_folds']}개 회차, {ho['holdout_rounds'][0]}~{ho['holdout_rounds'][1]}회)\n\n"
            "| 지표 | 값 | 기준 |\n|---|---|---|\n"
            f"| holdout edge (95% CI) | {ho['edge_nats']:+.4f} ({ho['edge_ci95'][0]:+.4f} ~ {ho['edge_ci95'][1]:+.4f}) | CI 하한 > 0 이어야 우위 |\n"
            f"| holdout 평균 적중 (95% CI) | {ho['mean_hits']:.3f} ({ho['mean_hits_ci95'][0]:.2f} ~ {ho['mean_hits_ci95'][1]:.2f}) | 무작위 0.800 |\n"
            f"| 적중 p 단측 / 양측 | {ho['hits_p_one_sided']:.3f} / {ho['hits_p_two_sided']:.3f} | 단측 < 0.05 |\n"
            f"| 균등 가중치 기준선 edge / 적중 | {b['edge_nats']:+.4f} / {b['mean_hits']:.3f} | 비교 대상 |\n"
            f"| 최적화 − 기준선 edge (95% CI) | {ho['ensemble_minus_equal_edge']:+.4f} ({d[0]:+.4f} ~ {d[1]:+.4f}) | 0 포함 시 차이 없음 |\n"
            f"| 감사 판정 | **{audit.get('verdict', '-')}** | |")
    for f in audit.get("risk_flags", []):
        print("⚠️", f)
    print("\n[최종 세트 상세] score = c_prob + c_pair + c_typical + c_prime")
    _show(res["final_set_details"][["set", "score_rank", "numbers", "score", "c_prob", "c_pair", "c_typical", "c_prime",
                                     "sum", "odd", "high", "ac", "zones_used"]].round(3))
    print("\n[전략별 백테스트 요약] (무작위 기대 적중 0.8)")
    _show(res["backtest_summary"][["strategy", "mean_hits", "ll_edge", "z_score", "p_value", "p_two_sided", "p_bonferroni"]].round(4))
    llm = ctx.artifacts.get("llm") or {}
    if llm.get("ok"):
        steps = llm.get("steps", {})
        _markdown(f"---\n## 🤖 AI 리뷰 ({llm['provider']} / `{llm['model']}`)\n\n### 1) 분석가\n\n"
                  + (steps.get("analyst", {}).get("content") or "-"))
        print("\n[2) 비평가]")
        print(json.dumps(llm.get("critic") or steps.get("critic", {}).get("content"), ensure_ascii=False, indent=1))
        print("\n[3) 최적화 자문 → 적용 가능한 제안]")
        print(json.dumps(llm.get("suggestions") or {}, ensure_ascii=False, indent=1))
        costs = [s.get("cost_usd") for s in steps.values() if s.get("cost_usd")]
        if costs:
            print(f"Claude 비용 합계: ${sum(costs):.4f}")
    elif llm:
        print("\nAI 리뷰:", llm.get("error"))
    print("\n[에이전트 실행 결과]")
    for a in ctx.agent_results:
        print(f"- {a.name}: {a.status} | {a.message}")


# 4) 실행
ctx = run(build_cfg(OUT_DIR / "run1"), "1차 실행")
report(ctx)

sugg = (ctx.artifacts.get("llm") or {}).get("suggestions")
if APPLY_AI_SUGGESTIONS and sugg:
    print("\n🔁 AI 자문 제안 적용:", json.dumps(sugg, ensure_ascii=False))
    ctx2 = run(apply_suggestions(build_cfg(OUT_DIR / "run2_ai_applied"), sugg), "2차 실행 (AI 자문 적용)")
    report(ctx2)
elif APPLY_AI_SUGGESTIONS:
    print("\nℹ️ 적용할 AI 자문 제안이 없어 2차 실행을 건너뜁니다 (LLM_PROVIDER 확인).")

# 5) 결과 저장/다운로드
if OUT_DIR.exists():
    archive = shutil.make_archive(str(Path.cwd() / "lotto_results"), "zip", OUT_DIR)
    print(f"\n💾 결과 저장: {OUT_DIR}  (압축: {archive})")
    if DOWNLOAD_RESULTS and IN_COLAB:
        from google.colab import files
        files.download(archive)
