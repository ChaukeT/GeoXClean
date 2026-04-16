from block_model_viewer.drillholes.compositing_engine import (
    BreakMode,
    CompositeConfig,
    CompositingMethod,
    CompositingMethodEngine,
    Interval,
    PartialStrategy,
    WeightingMode,
)


def test_fixed_length_composites_preserve_lithology_and_domain_metadata_with_hard_breaks():
    intervals = [
        Interval("DH001", 0.0, 1.0, grades={"ZN": 1.0}, lith="SANDSTONE", domain="SED"),
        Interval("DH001", 1.0, 2.0, grades={"ZN": 2.0}, lith="SANDSTONE", domain="SED"),
        Interval("DH001", 2.0, 3.0, grades={"ZN": 3.0}, lith="LIMESTONE", domain="CARB"),
        Interval("DH001", 3.0, 4.0, grades={"ZN": 4.0}, lith="LIMESTONE", domain="CARB"),
    ]
    cfg = CompositeConfig(
        method=CompositingMethod.FIXED_LENGTH,
        composite_length=2.0,
        weighting_mode=WeightingMode.LENGTH,
        partial_strategy=PartialStrategy.KEEP,
        break_mode=BreakMode.HARD,
        hard_break_lithology=True,
        hard_break_domain=True,
    )

    composites = CompositingMethodEngine().composite(intervals, cfg)

    assert len(composites) == 2
    assert composites[0].metadata["lith_code"] == "SANDSTONE"
    assert composites[0].metadata["domain"] == "SED"
    assert composites[1].metadata["lith_code"] == "LIMESTONE"
    assert composites[1].metadata["domain"] == "CARB"
    assert "mixed_lithology" not in composites[0].metadata
    assert "mixed_domain" not in composites[0].metadata
