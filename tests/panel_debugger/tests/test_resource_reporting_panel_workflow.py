import numpy as np
import pandas as pd
import pytest

from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.ui.resource_reporting_panel import ResourceReportingPanel


def _fresh_registry():
    existing = DataRegistry.get_existing()
    if existing is not None:
        try:
            existing.clear_all()
        except Exception:
            pass
    DataRegistry._instance = None
    return DataRegistry.instance()


@pytest.mark.integration
def test_resource_reporting_panel_lists_irbf_domain_from_registered_mask(mock_qapp):
    registry = _fresh_registry()
    block_df = pd.DataFrame(
        {
            "X": np.arange(8, dtype=float) * 10.0,
            "Y": np.zeros(8, dtype=float),
            "Z": np.zeros(8, dtype=float),
            "CU_EST": np.linspace(0.2, 1.6, 8),
            "CLASS_FINAL": ["Inferred"] * 8,
        }
    )

    registry.register_indicator_rbf_domain(
        {
            "domain_name": "Inside",
            "inside_mask_shared": [True, True, False, False, True, False, True, False],
            "sample_domain_column": "IRBF_Domain",
        },
        source_panel="resource-reporting-irbf-domain-test",
    )

    panel = ResourceReportingPanel()
    panel._init_registry()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_bm_generated(block_df)
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain" in combo_items
        assert panel.domain_combo.currentText() == "IRBF_Domain"
        assert panel.block_model_data is not None
        assert "IRBF_Domain" in panel.block_model_data.columns
        assert set(panel.block_model_data["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside", "Outside"}
    finally:
        panel.close()
        registry.clear_all()
        DataRegistry._instance = None


@pytest.mark.integration
def test_resource_reporting_panel_refreshes_irbf_domain_after_registration(mock_qapp):
    registry = _fresh_registry()
    block_df = pd.DataFrame(
        {
            "X": np.arange(8, dtype=float) * 10.0,
            "Y": np.zeros(8, dtype=float),
            "Z": np.zeros(8, dtype=float),
            "CU_EST": np.linspace(0.2, 1.6, 8),
            "CLASS_FINAL": ["Inferred"] * 8,
        }
    )

    panel = ResourceReportingPanel()
    panel._init_registry()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_bm_generated(block_df)
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain" not in combo_items

        registry.register_indicator_rbf_domain(
            {
                "domain_name": "Inside",
                "inside_mask_shared": [True, True, False, False, True, False, True, False],
                "sample_domain_column": "IRBF_Domain",
            },
            source_panel="resource-reporting-irbf-refresh-test",
        )
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain" in combo_items
        assert panel.block_model_data is not None
        assert "IRBF_Domain" in panel.block_model_data.columns
    finally:
        panel.close()
        registry.clear_all()
        DataRegistry._instance = None
