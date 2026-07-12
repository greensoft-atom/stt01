from courtstt.textnorm import normalize_for_cer


def test_strips_punctuation_and_whitespace():
    assert normalize_for_cer("판결을 선고합니다.") == "판결을선고합니다"
    assert normalize_for_cer("피고인은,  무죄!") == "피고인은무죄"


def test_lowercases_latin():
    assert normalize_for_cer("KTX 열차") == "ktx열차"


def test_spacing_differences_do_not_differ():
    assert normalize_for_cer("동백꽃 속으로") == normalize_for_cer("동백꽃속으로")


def test_nfc_composition():
    decomposed = "한"  # 한 as jamo sequence
    assert normalize_for_cer(decomposed) == "한"
