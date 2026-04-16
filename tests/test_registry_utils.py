import pandas as pd

from block_model_viewer.drillholes.registry_utils import (
    build_database_from_registry,
    composite_assays_from_df,
    composite_lithology_from_df,
)


def test_build_database_from_registry_accepts_bhid_at_brg_aliases():
    data = {
        "collars": pd.DataFrame(
            {
                "BHID": ["DH001"],
                "X": [1000.0],
                "Y": [2000.0],
                "Z": [300.0],
                "LENGTH": [50.0],
            }
        ),
        "surveys": pd.DataFrame(
            {
                "BHID": ["DH001", "DH001"],
                "AT": [0.0, 50.0],
                "DIP": [-90.0, -90.0],
                "BRG": [25.0, 25.0],
            }
        ),
        "assays": pd.DataFrame(
            {
                "BHID": ["DH001", "DH001"],
                "FROM": [0.0, 1.0],
                "TO": [1.0, 2.0],
                "ZN": [4.0, 5.0],
            }
        ),
    }

    db = build_database_from_registry(data)

    assert len(db.collars) == 1
    assert len(db.surveys) == 2
    assert len(db.assays) == 2
    assert db.surveys["azimuth"].tolist() == [25.0, 25.0]


def test_composite_registry_helpers_accept_bhid_aliases():
    composites_df = pd.DataFrame(
        {
            "BHID": ["DH001"],
            "FROM": [0.0],
            "TO": [2.0],
            "ZN": [6.5],
            "LITH": ["OX"],
        }
    )

    assays = composite_assays_from_df(composites_df)
    lithology = composite_lithology_from_df(composites_df)

    assert len(assays) == 1
    assert assays[0].hole_id == "DH001"
    assert assays[0].values["ZN"] == 6.5
    assert len(lithology) == 1
    assert lithology[0].lith_code == "OX"
