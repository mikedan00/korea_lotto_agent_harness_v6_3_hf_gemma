from src.data_loader import load_history


def test_load_history_default():
    # XLS(1225회) 파일이 없으면 CSV(1~1224회)만 로드된다.
    draws = load_history("data/korealotto_1224V_CSV2.csv", "data/LottoSpecial_공나온순서_1225.xls")
    assert len(draws) >= 1224
    assert draws[-1].round >= 1224
    assert len(draws[-1].numbers) == 6
