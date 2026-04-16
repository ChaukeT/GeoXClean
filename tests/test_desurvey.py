import numpy as np
import pandas as pd

from block_model_viewer.utils.desurvey import minimum_curvature_desurvey


def test_minimum_curvature_desurvey_supports_at_brg_and_positive_down_dip():
    survey = pd.DataFrame(
        {
            "AT": [0.0, 100.0],
            "BRG": [90.0, 90.0],
            "DIP": [45.0, 45.0],
        }
    )

    depths, xs, ys, zs = minimum_curvature_desurvey(
        collar_x=0.0,
        collar_y=0.0,
        collar_z=1000.0,
        survey_df=survey,
    )

    assert depths is not None
    assert np.isclose(depths[-1], 100.0)
    assert xs[-1] > 0.0
    assert abs(ys[-1]) < 1e-6
    assert zs[-1] < 1000.0
