import pandas as pd

from block_model_viewer.drillholes.compositing_utils import dataframes_to_intervals


def test_dataframes_to_intervals_accepts_bhid_aliases():
    assays_df = pd.DataFrame(
        {
            "BHID": ["DH001", "DH001"],
            "FROM": [0.0, 1.0],
            "TO": [1.0, 2.0],
            "ZN": [2.5, 3.5],
        }
    )
    lithology_df = pd.DataFrame(
        {
            "BHID": ["DH001", "DH001"],
            "FROM": [0.0, 1.0],
            "TO": [1.0, 2.0],
            "LITH": ["OX", "FR"],
        }
    )

    result = dataframes_to_intervals(assays_df=assays_df, lithology_df=lithology_df)

    assert result.rows_included == 2
    assert result.rows_excluded == 0
    assert [iv.hole_id for iv in result.intervals] == ["DH001", "DH001"]
    assert [iv.grades["ZN"] for iv in result.intervals] == [2.5, 3.5]
    assert [iv.lith for iv in result.intervals] == ["OX", "FR"]

