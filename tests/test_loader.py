from src.data_loader import load_history


def test_load_history_default():
    draws = load_history("data/korealotto_1224V_CSV2.csv", "data/LottoSpecial_공나온순서_1225.xls")
    assert len(draws) >= 1225
    assert draws[-1].round >= 1225
    assert len(draws[-1].numbers) == 6
