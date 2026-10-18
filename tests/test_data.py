import numpy as np
import pandas as pd

from heartrisk.data import FEATURES, RAW_COLUMNS, TARGET, clean, load_raw, split_xy


def _raw_rows() -> pd.DataFrame:
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],  # duplicate
        [41, 0, 2, 130, 0, 0, 2, 172, 0, 1.4, 1, np.nan, 3, 0],  # chol=0 invalid
        [56, 1, 9, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, np.nan, 4],  # cp=9 invalid code
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_load_raw_parses_question_marks(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,?,0\n")
    df = load_raw(f)
    assert list(df.columns) == RAW_COLUMNS
    assert df[["ca", "thal"]].isna().all(axis=None)


def test_clean_binarises_target():
    out = clean(_raw_rows())
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].tolist() == [0, 1, 0, 1]  # num>0 -> 1


def test_clean_drops_duplicates_and_keeps_schema():
    out = clean(_raw_rows())
    assert len(out) == 4
    assert list(out.columns) == FEATURES + [TARGET]
    assert "num" not in out.columns


def test_clean_marks_invalid_values_missing():
    out = clean(_raw_rows())
    assert np.isnan(out.loc[2, "chol"])  # 0 cholesterol is impossible
    assert np.isnan(out.loc[3, "cp"])  # unknown chest-pain code


def test_clean_does_not_mutate_input():
    raw = _raw_rows()
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_split_xy(synthetic_df):
    X, y = split_xy(synthetic_df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET and len(X) == len(y)
