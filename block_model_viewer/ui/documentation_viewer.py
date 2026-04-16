"""
Documentation Viewer — comprehensive help system for GeoX.

Opened via Help > Documentation (F1).
Provides a navigable tree of all GeoX features with embedded HTML content.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTextBrowser,
    QLineEdit, QPushButton, QLabel, QWidget,
)

from .modern_styles import ModernColors

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class DocumentationViewer(QDialog):
    """Professional documentation browser with sidebar navigation and search."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GeoX Documentation")
        self.resize(1050, 720)
        self.setMinimumSize(700, 500)

        self._content: Dict[str, Tuple[str, str]] = {}
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(150)
        self._search_timer.timeout.connect(self._apply_filter)

        self._setup_ui()
        self._build_content()
        self._build_nav_tree()

        # Select first topic
        first = self._nav_tree.topLevelItem(0)
        if first and first.childCount():
            first.setExpanded(True)
            self._nav_tree.setCurrentItem(first.child(0))

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # --- Search row ---
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search documentation...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)
        self._search_input.setStyleSheet(
            f"QLineEdit {{ padding: 8px; border: 1px solid {ModernColors.BORDER}; "
            f"border-radius: 4px; background: {ModernColors.ELEVATED_BG}; "
            f"color: {ModernColors.TEXT_PRIMARY}; }}"
        )
        search_row.addWidget(self._search_input)

        self._result_label = QLabel("")
        self._result_label.setStyleSheet(
            f"color: {ModernColors.TEXT_SECONDARY}; padding-right: 4px;"
        )
        self._result_label.setFixedWidth(80)
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        search_row.addWidget(self._result_label)
        root.addLayout(search_row)

        # --- Splitter (nav + content) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._nav_tree = QTreeWidget()
        self._nav_tree.setHeaderHidden(True)
        self._nav_tree.setMinimumWidth(220)
        self._nav_tree.setMaximumWidth(340)
        self._nav_tree.currentItemChanged.connect(self._on_item_changed)
        self._nav_tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background-color: {ModernColors.PANEL_BG};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 6px;
                padding: 4px;
            }}
            QTreeWidget::item {{
                padding: 4px 6px;
                border-radius: 3px;
            }}
            QTreeWidget::item:selected {{
                background-color: {ModernColors.ACCENT_PRIMARY};
                color: white;
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {ModernColors.CARD_HOVER};
            }}
            """
        )
        splitter.addWidget(self._nav_tree)

        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setReadOnly(True)
        self._browser.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: {ModernColors.CARD_BG};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER};
                border-radius: 8px;
            }}
            QScrollBar:vertical {{
                background: {ModernColors.PANEL_BG};
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {ModernColors.BORDER_LIGHT};
                min-height: 24px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ModernColors.ACCENT_PRIMARY};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
                height: 0px;
            }}
            """
        )
        splitter.addWidget(self._browser)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 780])
        root.addWidget(splitter, 1)

        # --- Footer ---
        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    # ------------------------------------------------------------------
    # Navigation tree
    # ------------------------------------------------------------------

    def _add_section(self, title: str, topics: list):
        section = QTreeWidgetItem(self._nav_tree, [title])
        section.setFlags(section.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        font = section.font(0)
        font.setBold(True)
        section.setFont(0, font)
        for topic_key, topic_title in topics:
            child = QTreeWidgetItem(section, [topic_title])
            child.setData(0, Qt.ItemDataRole.UserRole, topic_key)
        section.setExpanded(False)
        return section

    def _build_nav_tree(self):
        self._nav_tree.clear()

        self._add_section("Getting Started", [
            ("getting_started/welcome", "Welcome to GeoX"),
            ("getting_started/quickstart", "Quick Start Workflow"),
            ("getting_started/project", "Project Overview"),
        ])
        self._add_section("Data Import", [
            ("data_import/block_model", "Block Model Import"),
            ("data_import/file_formats", "CSV & File Formats"),
            ("data_import/column_mapping", "Column Mapping"),
        ])
        self._add_section("Drillhole Workflow", [
            ("drillholes/importing", "Importing Drillholes"),
            ("drillholes/validation", "Validation & QC"),
            ("drillholes/compositing", "Compositing"),
            ("drillholes/declustering", "Declustering"),
            ("drillholes/transformation", "Grade Transformation"),
        ])
        self._add_section("Geostatistics", [
            ("geostats/variogram", "Variogram Analysis"),
            ("geostats/variogram_assistant", "Variogram Assistant"),
            ("geostats/ok", "Ordinary Kriging"),
            ("geostats/sk", "Simple Kriging"),
            ("geostats/uk", "Universal Kriging"),
            ("geostats/ik", "Indicator Kriging"),
            ("geostats/cok", "Co-Kriging"),
            ("geostats/soft", "Soft Kriging"),
            ("geostats/bayesian", "Bayesian Kriging"),
            ("geostats/sgsim", "SGSIM"),
            ("geostats/cosgsim", "Co-SGSIM"),
            ("geostats/ik_sgsim", "IK-SGSIM"),
            ("geostats/sis", "SIS"),
            ("geostats/grf", "Gaussian Random Fields"),
            ("geostats/turning_bands", "Turning Bands"),
            ("geostats/mps", "Multiple Point Statistics"),
            ("geostats/dbs", "Direct Block Simulation"),
        ])
        self._add_section("Geological Modelling", [
            ("geology/loopstructural", "LoopStructural Integration"),
            ("geology/faults", "Faults"),
            ("geology/folds", "Folds"),
            ("geology/veins", "Veins"),
            ("geology/surfaces", "Surface Extraction"),
        ])
        self._add_section("Resource Classification", [
            ("resources/jorc", "JORC / SAMREC Classification"),
            ("resources/reporting", "Resource Reporting"),
            ("resources/grade_tonnage", "Grade-Tonnage Curves"),
            ("resources/block_calc", "Block Property Calculator"),
        ])
        self._add_section("Mine Planning", [
            ("planning/pit", "Pit Optimization"),
            ("planning/pushback", "Pushback Design"),
            ("planning/bench", "Bench Design"),
            ("planning/strategic", "Strategic Scheduling"),
            ("planning/tactical", "Tactical Scheduling"),
            ("planning/short_term", "Short-Term Scheduling"),
            ("planning/underground", "Underground Mining"),
            ("planning/fleet", "Fleet Management"),
            ("planning/economics", "NPV / IRR Economics"),
        ])
        self._add_section("Visualization", [
            ("vis/viewer", "3D Viewer Controls"),
            ("vis/display", "Display Settings"),
            ("vis/camera", "Camera & Navigation"),
            ("vis/mouse", "Mouse Modes"),
            ("vis/clip", "Clip Planes"),
            ("vis/legend", "Legends & Scale Bar"),
            ("vis/screenshot", "Screenshot Export"),
            ("vis/layout", "Layout Export"),
        ])
        self._add_section("Tools & Utilities", [
            ("tools/palette", "Command Palette"),
            ("tools/registry", "Data Registry"),
            ("tools/statistics", "Statistics Panel"),
            ("tools/charts", "Charts Panel"),
            ("tools/export", "Data Export"),
            ("tools/preferences", "Preferences"),
        ])
        self._add_section("Keyboard Shortcuts", [
            ("shortcuts/file", "File Operations"),
            ("shortcuts/view", "View Controls"),
            ("shortcuts/camera", "Camera Navigation"),
            ("shortcuts/bookmarks", "View Bookmarks"),
            ("shortcuts/display", "Display Toggles"),
            ("shortcuts/mouse", "Mouse Modes"),
        ])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search_changed(self, text: str):
        self._search_timer.start()

    def _apply_filter(self):
        query = self._search_input.text().lower().strip()
        match_count = 0

        for i in range(self._nav_tree.topLevelItemCount()):
            section = self._nav_tree.topLevelItem(i)
            section_visible = False

            for j in range(section.childCount()):
                child = section.child(j)
                key = child.data(0, Qt.ItemDataRole.UserRole)
                entry = self._content.get(key)
                if not entry:
                    child.setHidden(bool(query))
                    continue

                title, html = entry
                visible = (not query) or (query in title.lower()) or (
                    query in _HTML_TAG_RE.sub("", html).lower()
                )
                child.setHidden(not visible)
                if visible:
                    section_visible = True
                    match_count += 1

            section.setHidden(not section_visible)
            if section_visible and query:
                section.setExpanded(True)

        if query:
            self._result_label.setText(f"{match_count} results")
        else:
            self._result_label.setText("")

    # ------------------------------------------------------------------
    # Content display
    # ------------------------------------------------------------------

    def _on_item_changed(self, current, _previous):
        if current is None:
            return
        key = current.data(0, Qt.ItemDataRole.UserRole)
        if key and key in self._content:
            title, body = self._content[key]
            self._browser.setHtml(self._wrap_html(body))
            self._browser.verticalScrollBar().setValue(0)

    def _wrap_html(self, body: str) -> str:
        return f"""<html><head><style>
body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: {ModernColors.TEXT_PRIMARY};
    background-color: {ModernColors.CARD_BG};
    padding: 20px 28px;
    line-height: 1.65;
}}
h2 {{
    color: {ModernColors.ACCENT_PRIMARY};
    border-bottom: 2px solid {ModernColors.BORDER};
    padding-bottom: 8px;
    margin-top: 0;
    font-size: 20px;
}}
h3 {{
    color: {ModernColors.TEXT_PRIMARY};
    margin-top: 22px;
    font-size: 15px;
}}
h4 {{
    color: {ModernColors.TEXT_SECONDARY};
    margin-top: 16px;
    font-size: 13px;
}}
p {{ margin: 8px 0; }}
ul, ol {{ margin: 8px 0 8px 20px; }}
li {{ margin: 3px 0; }}
code {{
    background-color: {ModernColors.ELEVATED_BG};
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
pre {{
    background-color: {ModernColors.ELEVATED_BG};
    padding: 12px;
    border-radius: 6px;
    border: 1px solid {ModernColors.BORDER};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    overflow-x: auto;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}}
th, td {{
    padding: 8px 12px;
    border: 1px solid {ModernColors.BORDER};
    text-align: left;
}}
th {{
    background-color: {ModernColors.ELEVATED_BG};
    font-weight: 600;
}}
.tip {{
    background-color: {ModernColors.ELEVATED_BG};
    border-left: 4px solid {ModernColors.ACCENT_PRIMARY};
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
}}
.warning {{
    background-color: {ModernColors.ELEVATED_BG};
    border-left: 4px solid #e6a23c;
    padding: 10px 14px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
}}
kbd {{
    background-color: {ModernColors.ELEVATED_BG};
    border: 1px solid {ModernColors.BORDER};
    border-radius: 3px;
    padding: 2px 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}}
</style></head><body>{body}</body></html>"""

    # ------------------------------------------------------------------
    # Content registry
    # ------------------------------------------------------------------

    def _build_content(self):
        c = self._content

        # ==============================================================
        # GETTING STARTED
        # ==============================================================

        c["getting_started/welcome"] = ("Welcome to GeoX", """
<h2>Welcome to GeoX</h2>
<p>GeoX is a professional geoscience visualization and resource estimation platform
designed for mining engineers, geologists, and geostatisticians. It provides an
integrated environment for the complete mineral resource evaluation workflow &mdash;
from raw drillhole data to JORC/SAMREC-compliant resource reports.</p>

<h3>Key Capabilities</h3>
<ul>
<li><b>3D Visualization</b> &mdash; Interactive block model and drillhole visualization
powered by VTK/PyVista</li>
<li><b>Drillhole Management</b> &mdash; Import, validate, composite, and decluster
drillhole data with full QC workflows</li>
<li><b>Geostatistics</b> &mdash; Complete suite including variogram analysis, 7 kriging
methods, and 8 simulation algorithms</li>
<li><b>Geological Modelling</b> &mdash; LoopStructural-based implicit modelling with
fault, fold, and vein support</li>
<li><b>Resource Classification</b> &mdash; JORC/SAMREC-compliant classification,
grade-tonnage analysis, and resource reporting</li>
<li><b>Mine Planning</b> &mdash; Pit optimization, scheduling (strategic/tactical/short-term),
fleet management, and economic analysis</li>
<li><b>Geotechnical Analysis</b> &mdash; Slope stability, rockburst assessment,
stope stability, and risk analysis</li>
</ul>

<div class="tip"><b>Tip:</b> Press <kbd>Ctrl+Shift+P</kbd> at any time to open the
Command Palette and quickly search for any feature or panel.</div>
""")

        c["getting_started/quickstart"] = ("Quick Start Workflow", """
<h2>Quick Start Workflow</h2>
<p>Follow these steps to get from raw data to resource estimation:</p>

<h3>Step 1: Import Your Data</h3>
<p>Go to <b>File &gt; Open</b> (or <kbd>Ctrl+O</kbd>) to import a block model CSV
or VTK file. GeoX auto-detects coordinate columns and property fields.</p>
<p>For drillhole data, go to <b>Data &gt; Load Drillhole Data</b> to import
collar, survey, assay, and lithology tables.</p>

<h3>Step 2: Explore &amp; Validate</h3>
<p>Use the 3D viewer to visually inspect your data. Orbit with the mouse,
press <kbd>R</kbd> to reset the view, or use number keys <kbd>1</kbd>&ndash;<kbd>7</kbd>
for preset camera positions.</p>
<p>For drillholes, open <b>Data &gt; Drillhole QC &amp; Validation</b> to run
comprehensive quality checks and auto-fix common issues.</p>

<h3>Step 3: Process Data</h3>
<p>Composite drillhole samples to regular intervals via <b>Data &gt; Composite Drillholes</b>.
Decluster composites using <b>Modelling &gt; Declustering</b> for unbiased statistics.</p>

<h3>Step 4: Variogram Analysis</h3>
<p>Compute and fit variograms via <b>Modelling &gt; Variogram Analysis</b>. Use the
Variogram Assistant for semi-automatic model fitting with cross-validation.</p>

<h3>Step 5: Estimation / Simulation</h3>
<p>Run kriging (OK, SK, UK, IK, Co-K) or simulation (SGSIM, SIS, etc.) from the
<b>Modelling</b> menu. Configure search parameters, variogram model, and output
grid settings.</p>

<h3>Step 6: Classify &amp; Report</h3>
<p>Classify resources per JORC/SAMREC standards via <b>Resources &gt; JORC Classification</b>.
Generate grade-tonnage curves and resource tables for reporting.</p>

<div class="tip"><b>Tip:</b> Each panel has built-in tooltips. Hover over any
control for a brief explanation of its purpose.</div>
""")

        c["getting_started/project"] = ("Project Overview", """
<h2>Project Overview</h2>
<p>GeoX organises your work around a <b>Data Registry</b> &mdash; a central
repository that tracks all loaded datasets, their processing history, and
relationships between data sources.</p>

<h3>Supported File Formats</h3>
<table>
<tr><th>Format</th><th>Extension</th><th>Use Case</th></tr>
<tr><td>CSV / TSV</td><td>.csv, .tsv</td><td>Block models, drillhole tables, generic tabular data</td></tr>
<tr><td>VTK Legacy</td><td>.vtk</td><td>Structured/unstructured grids</td></tr>
<tr><td>VTK XML</td><td>.vtu, .vts, .vti, .vtr</td><td>Modern VTK formats</td></tr>
<tr><td>STL</td><td>.stl</td><td>Surface meshes</td></tr>
<tr><td>OBJ</td><td>.obj</td><td>3D surface export</td></tr>
<tr><td>Leapfrog</td><td>.aproj</td><td>Leapfrog Geo project import</td></tr>
</table>

<h3>Data Registry</h3>
<p>The Data Registry automatically tracks every dataset loaded or generated during
your session. You can inspect it via <b>Tools &gt; Data Registry Status</b>
(developer mode). Key concepts:</p>
<ul>
<li><b>Immutable drillhole data</b> &mdash; raw drillhole data is never modified;
all transformations create new datasets</li>
<li><b>Block model as authoritative source</b> &mdash; the block model is the
single source of truth for spatial properties</li>
<li><b>Provenance tracking</b> &mdash; every derived dataset references its parent
sources for full traceability</li>
</ul>
""")

        # ==============================================================
        # DATA IMPORT
        # ==============================================================

        c["data_import/block_model"] = ("Block Model Import", """
<h2>Block Model Import</h2>
<p>GeoX supports importing block models from CSV files with automatic column
detection, as well as VTK structured and unstructured grids.</p>

<h3>Importing a CSV Block Model</h3>
<ol>
<li>Go to <b>File &gt; Open</b> or press <kbd>Ctrl+O</kbd></li>
<li>Select your CSV file containing block model data</li>
<li>GeoX will auto-detect coordinate columns (X, Y, Z or Easting, Northing, Elevation)</li>
<li>The <b>Column Mapping Dialog</b> opens for you to confirm or adjust the mapping</li>
<li>Click <b>Import</b> to load the model into the 3D viewer</li>
</ol>

<h3>Required Columns</h3>
<p>At minimum, your CSV must contain three coordinate columns representing block
centroids. All additional columns are treated as block properties (grade, density,
classification, etc.).</p>

<table>
<tr><th>Column Type</th><th>Common Names (auto-detected)</th></tr>
<tr><td>X / Easting</td><td>X, XC, XCENTRE, XCENTROID, EASTING, EAST</td></tr>
<tr><td>Y / Northing</td><td>Y, YC, YCENTRE, YCENTROID, NORTHING, NORTH</td></tr>
<tr><td>Z / Elevation</td><td>Z, ZC, ZCENTRE, ZCENTROID, ELEVATION, RL</td></tr>
</table>

<h3>Importing VTK Files</h3>
<p>VTK files (.vtk, .vtu, .vts, .vti, .vtr) are loaded directly without a mapping
step. All cell data arrays become available as block properties.</p>

<div class="tip"><b>Tip:</b> For large block models (&gt;500,000 blocks), GeoX
automatically applies level-of-detail rendering to maintain interactive frame rates.</div>
""")

        c["data_import/file_formats"] = ("CSV & File Formats", """
<h2>CSV &amp; File Formats</h2>

<h3>CSV Requirements</h3>
<ul>
<li>Comma, semicolon, or tab delimited</li>
<li>First row must be a header row with column names</li>
<li>Numeric columns should use period (.) as decimal separator</li>
<li>Missing values can be blank, NaN, or -999</li>
<li>UTF-8 encoding recommended (Latin-1 also supported)</li>
</ul>

<h3>Example CSV Format</h3>
<pre>XC,YC,ZC,AU_PPM,CU_PCT,DENSITY,LITH
500100.5,7200050.5,350.0,2.45,0.89,2.71,GRANITE
500110.5,7200050.5,350.0,1.82,0.56,2.68,GRANITE
500120.5,7200050.5,350.0,3.11,1.23,2.74,DIORITE</pre>

<h3>VTK Formats</h3>
<table>
<tr><th>Format</th><th>Description</th><th>Best For</th></tr>
<tr><td>.vtk</td><td>VTK Legacy format</td><td>General purpose, widely supported</td></tr>
<tr><td>.vtu</td><td>Unstructured Grid XML</td><td>Irregular block models</td></tr>
<tr><td>.vts</td><td>Structured Grid XML</td><td>Regular grids with rotation</td></tr>
<tr><td>.vti</td><td>Image Data XML</td><td>Regular grids (fastest)</td></tr>
<tr><td>.vtr</td><td>Rectilinear Grid XML</td><td>Variable-size regular blocks</td></tr>
</table>
""")

        c["data_import/column_mapping"] = ("Column Mapping", """
<h2>Column Mapping</h2>
<p>When importing a CSV block model, GeoX opens the Column Mapping Dialog to
confirm how your file columns map to block model properties.</p>

<h3>Auto-Detection</h3>
<p>GeoX uses pattern matching to automatically assign columns:</p>
<ul>
<li><b>Coordinate columns</b> &mdash; matched by name (X, XC, EASTING, etc.) and
by value range (large numeric values suggest UTM coordinates)</li>
<li><b>Grade columns</b> &mdash; names containing AU, CU, FE, AL, etc.</li>
<li><b>Classification columns</b> &mdash; names containing CLASS, CATEGORY, DOMAIN</li>
<li><b>Density columns</b> &mdash; names containing DENS, SG, BD</li>
</ul>

<h3>Manual Override</h3>
<p>If auto-detection makes an incorrect assignment, use the dropdown menus to
manually reassign any column. The preview table shows sample values to help
verify correctness.</p>

<h3>Block Size Detection</h3>
<p>GeoX infers block dimensions by analysing the spacing between coordinate
values. If your blocks have irregular sizes, they will be loaded as an
unstructured grid.</p>
""")

        # ==============================================================
        # DRILLHOLES
        # ==============================================================

        c["drillholes/importing"] = ("Importing Drillholes", """
<h2>Importing Drillholes</h2>
<p>Drillhole data in GeoX follows the standard four-table structure used across
the mining industry.</p>

<h3>Required Tables</h3>
<table>
<tr><th>Table</th><th>Required Columns</th><th>Description</th></tr>
<tr><td><b>Collar</b></td><td>HOLE_ID, X, Y, Z, MAX_DEPTH</td><td>Drillhole collar locations and total depths</td></tr>
<tr><td><b>Survey</b></td><td>HOLE_ID, DEPTH, AZIMUTH, DIP</td><td>Downhole deviation survey data</td></tr>
<tr><td><b>Assay</b></td><td>HOLE_ID, FROM, TO, &lt;grades&gt;</td><td>Geochemical assay intervals</td></tr>
<tr><td><b>Lithology</b></td><td>HOLE_ID, FROM, TO, LITH_CODE</td><td>Geological lithology logging (optional)</td></tr>
</table>

<h3>Import Steps</h3>
<ol>
<li>Go to <b>Data &gt; Load Drillhole Data</b></li>
<li>Select each table file (CSV format)</li>
<li>Map columns to their roles (auto-detection assists)</li>
<li>Click <b>Import</b> to load and desurvey the drillhole traces</li>
</ol>

<h3>Leapfrog Import</h3>
<p>GeoX can also import drillhole data directly from Leapfrog Geo project files
(.aproj). Go to <b>Data &gt; Import Leapfrog Geo Project</b>.</p>

<div class="tip"><b>Tip:</b> Drillhole data in GeoX is immutable. All processing
operations (compositing, declustering, transformation) create new derived datasets,
preserving your raw data intact.</div>
""")

        c["drillholes/validation"] = ("Validation & QC", """
<h2>Validation &amp; QC</h2>
<p>Quality control is essential before any geostatistical analysis. GeoX provides
comprehensive validation checks via <b>Data &gt; Drillhole QC &amp; Validation</b>.</p>

<h3>Validation Checks</h3>
<table>
<tr><th>Check</th><th>Description</th></tr>
<tr><td>Collar consistency</td><td>Verifies collar coordinates are within expected bounds</td></tr>
<tr><td>Survey integrity</td><td>Checks for missing surveys, impossible angles, depth mismatches</td></tr>
<tr><td>Interval overlaps</td><td>Detects overlapping FROM-TO intervals in assay/lithology tables</td></tr>
<tr><td>Interval gaps</td><td>Identifies unsampled intervals that may indicate data loss</td></tr>
<tr><td>Negative values</td><td>Flags negative assay grades (usually errors)</td></tr>
<tr><td>Duplicate holes</td><td>Finds holes with identical collar coordinates</td></tr>
<tr><td>Orphan records</td><td>Assay/survey records with no matching collar entry</td></tr>
</table>

<h3>Auto-Fix</h3>
<p>The auto-fix feature can automatically resolve common issues:</p>
<ul>
<li>Remove duplicate intervals</li>
<li>Fill small gaps with interpolated values</li>
<li>Correct negative assay values to zero</li>
<li>Remove orphan records</li>
</ul>

<div class="warning"><b>Warning:</b> Always review auto-fix results before proceeding.
Automated corrections are logged for audit trail compliance.</div>
""")

        c["drillholes/compositing"] = ("Compositing", """
<h2>Compositing</h2>
<p>Compositing regularises drillhole samples to uniform intervals, which is
essential for unbiased geostatistical analysis. Access via
<b>Data &gt; Composite Drillholes</b>.</p>

<h3>Compositing Methods</h3>
<table>
<tr><th>Method</th><th>Description</th><th>When to Use</th></tr>
<tr><td><b>Length-weighted</b></td><td>Combines samples proportional to their length
within the composite interval</td><td>Standard method for most deposits</td></tr>
<tr><td><b>Bench compositing</b></td><td>Composites to mining bench heights</td>
<td>When planning is by fixed bench levels</td></tr>
<tr><td><b>Fixed-length</b></td><td>Composites to a fixed downhole length</td>
<td>When uniform support is required</td></tr>
</table>

<h3>Parameters</h3>
<ul>
<li><b>Composite length</b> &mdash; target interval length (typically matches bench height)</li>
<li><b>Minimum length fraction</b> &mdash; minimum acceptable composite length as a
fraction of target (e.g., 0.5 means composites shorter than 50% of target are discarded)</li>
<li><b>Grade columns</b> &mdash; which assay columns to composite</li>
</ul>

<div class="tip"><b>Tip:</b> A common composite length is the bench height (e.g., 5m
or 10m). This ensures composites are directly comparable to block dimensions.</div>
""")

        c["drillholes/declustering"] = ("Declustering", """
<h2>Declustering</h2>
<p>Drillhole samples are typically clustered in areas of high grade or geological
interest. Declustering assigns weights to correct for this spatial bias.
Access via <b>Modelling &gt; Declustering</b>.</p>

<h3>Cell Declustering Method</h3>
<p>GeoX uses cell-based declustering, which is the standard JORC/SAMREC-defensible
approach:</p>
<ol>
<li>A grid of cells is overlaid on the sample locations</li>
<li>Each cell's weight is inversely proportional to the number of samples it contains</li>
<li>Weights are normalised so the sum equals the total number of samples</li>
<li>The process is repeated for multiple cell sizes to find the optimal size</li>
</ol>

<h3>Parameters</h3>
<ul>
<li><b>Minimum cell size</b> &mdash; smallest cell dimension to test</li>
<li><b>Maximum cell size</b> &mdash; largest cell dimension to test</li>
<li><b>Number of sizes</b> &mdash; how many sizes to evaluate between min and max</li>
<li><b>Optimal criterion</b> &mdash; minimum declustered mean indicates optimal cell size</li>
</ul>

<h3>Output</h3>
<p>Declustering produces a weight column that should be used in all subsequent
statistical analysis (histograms, variograms, etc.) to ensure unbiased estimates.</p>
""")

        c["drillholes/transformation"] = ("Grade Transformation", """
<h2>Grade Transformation</h2>
<p>Many geostatistical methods assume normally distributed data. Grade transformation
converts skewed raw data to a Gaussian distribution. Access via
<b>Modelling &gt; Grade Transformation</b>.</p>

<h3>Available Transforms</h3>
<table>
<tr><th>Transform</th><th>Formula</th><th>Use Case</th></tr>
<tr><td><b>Log transform</b></td><td>Y = ln(X)</td><td>Positively skewed data (common for gold, copper)</td></tr>
<tr><td><b>Square root</b></td><td>Y = &radic;X</td><td>Mildly skewed data</td></tr>
<tr><td><b>Box-Cox</b></td><td>Y = (X^&lambda; - 1) / &lambda;</td><td>Optimal power transform (auto-fitted)</td></tr>
<tr><td><b>Normal score</b></td><td>Quantile &rarr; Gaussian</td><td>Required for SGSIM and most simulations</td></tr>
</table>

<h3>Back-Transformation</h3>
<p>After estimation in transformed space, results are automatically back-transformed
to original units. For lognormal kriging, the back-transform includes the
appropriate bias correction.</p>

<div class="warning"><b>Warning:</b> Normal score transformation requires at least
30 samples for stable quantile mapping. Fewer samples may produce unreliable transforms.</div>
""")

        # ==============================================================
        # GEOSTATISTICS
        # ==============================================================

        c["geostats/variogram"] = ("Variogram Analysis", """
<h2>Variogram Analysis</h2>
<p>The variogram is the fundamental tool of geostatistics. It quantifies how
spatial correlation changes with distance and direction. Access via
<b>Modelling &gt; Variogram Analysis</b>.</p>

<h3>Experimental Variogram</h3>
<p>The experimental variogram is computed from sample pairs:</p>
<ul>
<li><b>Lag distance</b> &mdash; separation distance between pairs (typically half the average sample spacing)</li>
<li><b>Lag tolerance</b> &mdash; acceptable deviation from nominal lag (typically 50% of lag distance)</li>
<li><b>Number of lags</b> &mdash; how many distance bins to compute (typically 10&ndash;15)</li>
<li><b>Azimuth / Dip</b> &mdash; direction for directional variograms</li>
<li><b>Angular tolerance</b> &mdash; cone angle for directional search (typically 22.5&deg;)</li>
</ul>

<h3>Variogram Models</h3>
<table>
<tr><th>Model</th><th>Description</th></tr>
<tr><td><b>Spherical</b></td><td>Most common; reaches sill at finite range. Standard choice for most deposits.</td></tr>
<tr><td><b>Exponential</b></td><td>Approaches sill asymptotically. Suitable for gradational transitions.</td></tr>
<tr><td><b>Gaussian</b></td><td>Very smooth near origin. Use for highly continuous phenomena (rare in mining).</td></tr>
</table>

<h3>Key Parameters</h3>
<ul>
<li><b>Nugget (C0)</b> &mdash; variance at zero distance (measurement error + micro-scale variability)</li>
<li><b>Sill (C)</b> &mdash; total variance the model reaches</li>
<li><b>Range (a)</b> &mdash; distance at which spatial correlation effectively disappears</li>
</ul>
""")

        c["geostats/variogram_assistant"] = ("Variogram Assistant", """
<h2>Variogram Assistant</h2>
<p>The Variogram Assistant provides semi-automatic variogram fitting with
built-in cross-validation. Access via <b>Modelling &gt; Variogram Assistant</b>.</p>

<h3>Features</h3>
<ul>
<li><b>Automatic parameter suggestion</b> &mdash; analyses the experimental variogram
and proposes initial nugget, sill, and range values</li>
<li><b>Model comparison</b> &mdash; fits spherical, exponential, and Gaussian models
and ranks them by goodness-of-fit</li>
<li><b>Cross-validation</b> &mdash; leave-one-out cross-validation to assess the
quality of the fitted model</li>
<li><b>Quality gates</b> &mdash; warns about potential issues (insufficient pairs,
poor model fit, anisotropy inconsistencies)</li>
</ul>

<h3>Workflow</h3>
<ol>
<li>Select the variable and composited data source</li>
<li>The assistant computes directional variograms in major/minor/vertical directions</li>
<li>Review the suggested anisotropy ratios and ranges</li>
<li>Accept or adjust the fitted model</li>
<li>Run cross-validation to verify model quality</li>
</ol>
""")

        c["geostats/ok"] = ("Ordinary Kriging", """
<h2>Ordinary Kriging (OK)</h2>
<p>Ordinary Kriging is the most commonly used geostatistical estimation method.
It provides the Best Linear Unbiased Estimator (BLUE) assuming an unknown but
locally constant mean. Access via <b>Modelling &gt; Ordinary Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Standard resource estimation for most deposits</li>
<li>When the local mean is unknown or varies across the deposit</li>
<li>JORC/SAMREC-compliant resource estimation</li>
</ul>

<h3>Parameters</h3>
<table>
<tr><th>Parameter</th><th>Description</th></tr>
<tr><td>Variogram model</td><td>Fitted variogram (from Variogram Analysis)</td></tr>
<tr><td>Search radius</td><td>Maximum distance to search for samples</td></tr>
<tr><td>Min / Max samples</td><td>Minimum and maximum data points per block estimate</td></tr>
<tr><td>Block size</td><td>Output block dimensions (X, Y, Z increments)</td></tr>
<tr><td>Grid origin</td><td>Starting coordinates of the estimation grid</td></tr>
<tr><td>Grid extent</td><td>Number of blocks in each direction</td></tr>
</table>

<h3>Outputs</h3>
<ul>
<li><b>Estimated grade</b> &mdash; the kriged block values</li>
<li><b>Kriging variance</b> &mdash; estimation uncertainty for each block</li>
<li><b>Number of samples</b> &mdash; count of data points used per block</li>
</ul>
""")

        c["geostats/sk"] = ("Simple Kriging", """
<h2>Simple Kriging (SK)</h2>
<p>Simple Kriging assumes a known and constant global mean. It is used when
the mean grade is confidently established. Access via
<b>Modelling &gt; Simple Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Change-of-support corrections (e.g., uniform conditioning)</li>
<li>When the global mean is confidently known (e.g., from a large dataset)</li>
<li>As a component of SK cross-validation for stationarity checks</li>
</ul>

<h3>Key Difference from OK</h3>
<p>SK requires you to specify the global mean. In data-sparse areas, SK estimates
revert to this mean rather than to the local data average. This makes SK more
stable in extrapolation but requires a well-established mean value.</p>

<h3>SK Stationarity Check</h3>
<p>GeoX includes an SK stationarity analysis tool that computes the SK mean
across moving windows to test whether the stationarity assumption holds.
Access via <b>Modelling &gt; SK Stationarity</b>.</p>
""")

        c["geostats/uk"] = ("Universal Kriging", """
<h2>Universal Kriging (UK)</h2>
<p>Universal Kriging accounts for a deterministic trend in the data by estimating
both the trend and residual simultaneously. Access via
<b>Modelling &gt; Universal Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>When there is a clear spatial trend in the data (e.g., grade increases with depth)</li>
<li>Deposits with strong directional gradients</li>
</ul>

<h3>Drift Functions</h3>
<p>UK allows you to specify polynomial drift functions:</p>
<ul>
<li><b>Linear drift</b> &mdash; trend varies linearly with coordinates</li>
<li><b>Quadratic drift</b> &mdash; trend includes second-order terms</li>
<li><b>Custom drift</b> &mdash; user-specified external drift variables</li>
</ul>

<div class="warning"><b>Warning:</b> UK requires more data than OK to estimate
trend parameters reliably. Ensure adequate sample density before using UK.</div>
""")

        c["geostats/ik"] = ("Indicator Kriging", """
<h2>Indicator Kriging (IK)</h2>
<p>Indicator Kriging is a non-parametric method that estimates the probability
of exceeding specified thresholds. It is particularly useful for skewed
distributions. Access via <b>Modelling &gt; Indicator Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Highly skewed grade distributions (gold, platinum, diamonds)</li>
<li>When you need a full conditional CDF for each block</li>
<li>Cut-off grade sensitivity analysis</li>
<li>Risk assessment and probability mapping</li>
</ul>

<h3>Parameters</h3>
<ul>
<li><b>Thresholds</b> &mdash; cutoff values at which indicators are defined
(typically 5&ndash;9 thresholds spanning the grade range)</li>
<li><b>Indicator variograms</b> &mdash; separate variogram model for each threshold</li>
<li><b>Median IK option</b> &mdash; use a single variogram (at the median) for all thresholds</li>
</ul>

<h3>Output</h3>
<p>IK produces a conditional CDF for each block. From this CDF, GeoX computes:</p>
<ul>
<li>E-type estimate (conditional expectation)</li>
<li>Probability of exceeding each threshold</li>
<li>Conditional variance</li>
</ul>
""")

        c["geostats/cok"] = ("Co-Kriging", """
<h2>Co-Kriging</h2>
<p>Co-Kriging uses secondary variables (correlated with the primary variable of
interest) to improve estimation quality. Access via
<b>Modelling &gt; Co-Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>When a well-sampled secondary variable is correlated with a sparsely sampled
primary variable</li>
<li>Multi-element estimation where cross-correlations matter</li>
</ul>

<h3>Requirements</h3>
<ul>
<li>Direct variogram for the primary variable</li>
<li>Direct variogram for the secondary variable</li>
<li>Cross-variogram between the two variables</li>
<li>The linear model of coregionalisation (LMC) must be valid</li>
</ul>

<div class="tip"><b>Tip:</b> Co-Kriging is most beneficial when the secondary
variable has significantly more samples than the primary variable and the
cross-correlation is strong (|r| &gt; 0.5).</div>
""")

        c["geostats/soft"] = ("Soft Kriging", """
<h2>Soft Kriging</h2>
<p>Soft Kriging incorporates "soft" or uncertain data alongside hard data.
Soft data may come from geological interpretations, geophysical surveys, or
neighbouring deposits. Access via <b>Modelling &gt; Soft Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>When you have additional data sources with different levels of reliability</li>
<li>Integrating geological knowledge as soft constraints</li>
<li>Combining exploration data with production data of different quality</li>
</ul>

<h3>Soft Data Types</h3>
<ul>
<li><b>Interval constraints</b> &mdash; grade is known to lie within a range</li>
<li><b>Probability constraints</b> &mdash; probability of exceeding a threshold</li>
<li><b>Uncertain measurements</b> &mdash; measured values with known error distributions</li>
</ul>
""")

        c["geostats/bayesian"] = ("Bayesian Kriging", """
<h2>Bayesian Kriging</h2>
<p>Bayesian Kriging incorporates prior distributions on the variogram parameters
and provides posterior uncertainty estimates that account for both estimation
and model uncertainty. Access via <b>Modelling &gt; Bayesian Kriging</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Early-stage projects with limited data</li>
<li>When variogram parameter uncertainty is significant</li>
<li>For robust uncertainty quantification</li>
</ul>

<h3>Parameters</h3>
<ul>
<li><b>Prior distributions</b> &mdash; specified for nugget, sill, and range</li>
<li><b>MCMC iterations</b> &mdash; number of posterior samples to draw</li>
<li><b>Burn-in</b> &mdash; initial samples to discard for convergence</li>
</ul>

<h3>Output</h3>
<p>In addition to the standard kriged estimate, Bayesian Kriging produces
posterior distributions for all variogram parameters and a more realistic
uncertainty estimate for each block.</p>
""")

        c["geostats/sgsim"] = ("SGSIM", """
<h2>Sequential Gaussian Simulation (SGSIM)</h2>
<p>SGSIM generates multiple equiprobable realisations of the grade distribution
that honour the data values and reproduce the variogram model. It is the
standard simulation method in mining geostatistics. Access via
<b>Modelling &gt; SGSIM</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Uncertainty quantification &mdash; generating confidence intervals for tonnage and grade</li>
<li>Risk analysis &mdash; assessing probability of meeting production targets</li>
<li>Conditional simulation for mine planning</li>
</ul>

<h3>Parameters</h3>
<table>
<tr><th>Parameter</th><th>Description</th></tr>
<tr><td>Number of realisations</td><td>How many simulated fields to generate (typically 50&ndash;200)</td></tr>
<tr><td>Variogram model</td><td>Fitted variogram for the normal-score transformed data</td></tr>
<tr><td>Search parameters</td><td>Same as kriging (radius, min/max samples)</td></tr>
<tr><td>Random seed</td><td>For reproducibility of results</td></tr>
<tr><td>Grid specification</td><td>Output grid origin, extent, and block size</td></tr>
</table>

<h3>Output</h3>
<p>Each realisation is a complete block model. GeoX computes summary statistics
across realisations: E-type (mean), P10, P50, P90, and conditional variance.</p>

<div class="tip"><b>Tip:</b> Data must be normal-score transformed before SGSIM.
Use <b>Modelling &gt; Grade Transformation &gt; Normal Score</b> first.</div>
""")

        c["geostats/cosgsim"] = ("Co-SGSIM", """
<h2>Co-Sequential Gaussian Simulation (Co-SGSIM)</h2>
<p>Co-SGSIM simulates multiple correlated variables simultaneously, preserving
their cross-correlation structure. Access via <b>Modelling &gt; Co-SGSIM</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Multi-element deposits where element ratios matter</li>
<li>When co-kriging relationships are important for downstream processing</li>
</ul>

<h3>Requirements</h3>
<ul>
<li>Normal-score transformed data for all variables</li>
<li>Full coregionalisation model (direct + cross-variograms)</li>
<li>LMC must be positive definite</li>
</ul>
""")

        c["geostats/ik_sgsim"] = ("IK-SGSIM", """
<h2>Indicator Kriging SGSIM (IK-SGSIM)</h2>
<p>IK-SGSIM combines indicator kriging with sequential simulation to generate
realisations that respect indicator-based probability models. Access via
<b>Modelling &gt; IK-SGSIM</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Highly skewed grade distributions where Gaussian assumptions are poor</li>
<li>When indicator variograms capture grade continuity better than a single variogram</li>
</ul>
""")

        c["geostats/sis"] = ("SIS", """
<h2>Sequential Indicator Simulation (SIS)</h2>
<p>SIS simulates categorical variables (lithology, rock type, domains) by
treating each category as an indicator variable. Access via
<b>Modelling &gt; SIS</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Geological domain modelling &mdash; simulating lithological boundaries</li>
<li>Rock type probability mapping</li>
<li>Uncertainty in geological boundaries</li>
</ul>

<h3>Parameters</h3>
<ul>
<li><b>Categories</b> &mdash; the distinct rock types or domains to simulate</li>
<li><b>Indicator variograms</b> &mdash; one variogram per category indicator</li>
<li><b>Proportions</b> &mdash; global proportions of each category</li>
</ul>
""")

        c["geostats/grf"] = ("Gaussian Random Fields", """
<h2>Gaussian Random Fields (GRF)</h2>
<p>GRF generates unconditional Gaussian random fields using spectral methods.
Access via <b>Modelling &gt; GRF</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Generating synthetic reference models for testing</li>
<li>Monte Carlo uncertainty analysis</li>
<li>Stochastic background fields</li>
</ul>

<h3>Method</h3>
<p>GRF uses FFT-based spectral simulation, which is computationally efficient
for regular grids. The generated field honours the specified variogram model.</p>
""")

        c["geostats/turning_bands"] = ("Turning Bands", """
<h2>Turning Bands Simulation</h2>
<p>Turning Bands is an efficient simulation algorithm that generates 3D Gaussian
fields by combining 1D simulations along random lines (bands). Access via
<b>Modelling &gt; Turning Bands</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Large grids where SGSIM is too slow</li>
<li>When computational efficiency is a priority</li>
</ul>

<h3>Parameters</h3>
<ul>
<li><b>Number of bands</b> &mdash; more bands = better reproduction of target variogram
(typically 1000&ndash;2000)</li>
<li><b>Variogram model</b> &mdash; target covariance model</li>
</ul>

<div class="tip"><b>Tip:</b> Turning Bands is generally faster than SGSIM for large
grids (&gt;1 million cells) but may show banding artefacts with too few bands.</div>
""")

        c["geostats/mps"] = ("Multiple Point Statistics", """
<h2>Multiple Point Statistics (MPS)</h2>
<p>MPS uses training images to capture complex geological patterns that
two-point statistics (variograms) cannot represent. Access via
<b>Modelling &gt; MPS</b>.</p>

<h3>When to Use</h3>
<ul>
<li>Complex geological features (channels, curvilinear structures)</li>
<li>When variogram-based methods fail to reproduce geological patterns</li>
<li>Categorical simulation with complex spatial relationships</li>
</ul>

<h3>Requirements</h3>
<ul>
<li><b>Training image</b> &mdash; a 2D or 3D conceptual model of the expected geological
pattern (can be from outcrop mapping, geological interpretation, or prior models)</li>
<li><b>Conditioning data</b> &mdash; hard data points to honour</li>
</ul>
""")

        c["geostats/dbs"] = ("Direct Block Simulation", """
<h2>Direct Block Simulation (DBS)</h2>
<p>DBS simulates block-support values directly, avoiding the need for
point-to-block change-of-support corrections. Access via
<b>Modelling &gt; DBS</b>.</p>

<h3>When to Use</h3>
<ul>
<li>When block-support distributions are needed directly</li>
<li>To avoid change-of-support modelling errors</li>
<li>Grade control applications where block values are more relevant than point values</li>
</ul>

<h3>Advantages over Point Simulation + Block Averaging</h3>
<ul>
<li>Directly reproduces the block-support histogram</li>
<li>Avoids information effect smoothing</li>
<li>Computationally more efficient for large blocks</li>
</ul>
""")

        # ==============================================================
        # GEOLOGICAL MODELLING
        # ==============================================================

        c["geology/loopstructural"] = ("LoopStructural Integration", """
<h2>LoopStructural Integration</h2>
<p>GeoX integrates <b>LoopStructural</b>, an open-source implicit geological
modelling engine, for building 3D geological models from structural data.
Access via <b>Modelling &gt; Geological Model</b>.</p>

<h3>Capabilities</h3>
<ul>
<li>Implicit surface modelling using finite-difference interpolation (FDI)</li>
<li>Fault network modelling with displacement fields</li>
<li>Fold modelling with axial surface constraints</li>
<li>Unconformity handling</li>
<li>JORC/SAMREC compliance through standardised modelling workflows</li>
</ul>

<h3>Workflow</h3>
<ol>
<li>Load structural data (contacts, orientations, fault traces)</li>
<li>Define the model boundary (bounding box)</li>
<li>Add geological features: stratigraphy, faults, folds, unconformities</li>
<li>Build the model &mdash; LoopStructural solves the implicit functions</li>
<li>Extract surfaces and solids for visualization and resource domaining</li>
</ol>

<h3>Build Quality</h3>
<p>After building, review the <b>Build Log</b> and <b>Misfit Report</b> to
assess model quality. High misfits indicate the model does not honour the
input data well and may need parameter adjustment.</p>
""")

        c["geology/faults"] = ("Faults", """
<h2>Fault Definition</h2>
<p>Faults are modelled as discontinuities that offset geological surfaces.
Access via the <b>Fault Definition Panel</b> within Geological Modelling.</p>

<h3>Defining a Fault</h3>
<ul>
<li><b>Fault trace</b> &mdash; digitise or import the fault surface trace</li>
<li><b>Dip / Dip direction</b> &mdash; orientation of the fault plane</li>
<li><b>Displacement</b> &mdash; magnitude and direction of offset</li>
<li><b>Influence zone</b> &mdash; distance over which the fault affects surrounding rocks</li>
</ul>

<h3>Fault Interactions</h3>
<p>When multiple faults are present, specify their relative timing (which faults
cut which). LoopStructural handles the cross-cutting relationships automatically.</p>
""")

        c["geology/folds"] = ("Folds", """
<h2>Fold Definition</h2>
<p>Folds are modelled by constraining the foliation and axial surface geometry.
Access via the <b>Fold Definition Panel</b>.</p>

<h3>Parameters</h3>
<ul>
<li><b>Fold axis</b> &mdash; orientation (plunge and trend) of the fold hinge</li>
<li><b>Axial surface</b> &mdash; orientation of the axial plane</li>
<li><b>Wavelength</b> &mdash; distance between fold hinges</li>
<li><b>Fold type</b> &mdash; anticline, syncline, or more complex forms</li>
</ul>
""")

        c["geology/veins"] = ("Veins", """
<h2>Vein Definition</h2>
<p>Veins are modelled as tabular bodies with specified orientation and thickness.
Access via the <b>Vein Definition Panel</b>.</p>

<h3>Parameters</h3>
<ul>
<li><b>Centre surface</b> &mdash; the median surface of the vein</li>
<li><b>Thickness</b> &mdash; true thickness or apparent thickness with correction</li>
<li><b>Orientation</b> &mdash; strike, dip, and dip direction</li>
<li><b>Grade model</b> &mdash; optional: grade distribution within the vein</li>
</ul>
""")

        c["geology/surfaces"] = ("Surface Extraction", """
<h2>Surface Extraction</h2>
<p>After building a geological model, extract isosurfaces for visualization
and domain boundary definition.</p>

<h3>Extraction Options</h3>
<ul>
<li><b>Isosurface extraction</b> &mdash; extract surfaces at specified scalar values</li>
<li><b>Domain solids</b> &mdash; extract closed volumes for each geological domain</li>
<li><b>Fault surfaces</b> &mdash; extract fault planes as meshes</li>
</ul>

<h3>Export</h3>
<p>Extracted surfaces can be exported as:</p>
<ul>
<li>STL files for CAD/mine planning software</li>
<li>OBJ files for 3D visualisation tools</li>
<li>VTK files for further processing</li>
</ul>
""")

        # ==============================================================
        # RESOURCE CLASSIFICATION
        # ==============================================================

        c["resources/jorc"] = ("JORC / SAMREC Classification", """
<h2>JORC / SAMREC Resource Classification</h2>
<p>GeoX implements resource classification following the JORC Code (2012) and
SAMREC Code standards. Access via <b>Resources &gt; JORC Classification</b>.</p>

<h3>Classification Categories</h3>
<table>
<tr><th>Category</th><th>Criteria</th></tr>
<tr><td><b>Measured</b></td><td>High confidence: closely spaced data, well-understood geology,
kriging variance below threshold</td></tr>
<tr><td><b>Indicated</b></td><td>Reasonable confidence: moderate data spacing, adequate geological
understanding</td></tr>
<tr><td><b>Inferred</b></td><td>Low confidence: limited data, geological continuity assumed but
not confirmed</td></tr>
</table>

<h3>Classification Methods</h3>
<ul>
<li><b>Distance-based</b> &mdash; based on average distance to nearest drillholes</li>
<li><b>Kriging variance-based</b> &mdash; using estimation quality metrics</li>
<li><b>Number of samples</b> &mdash; minimum data support requirements</li>
<li><b>Slope of regression</b> &mdash; conditional bias indicator</li>
</ul>

<h3>Workflow</h3>
<ol>
<li>Select the estimated block model and drillhole data</li>
<li>Configure classification criteria (distance thresholds, variance limits)</li>
<li>Run classification &mdash; each block receives a category</li>
<li>Review classification statistics and spatial distribution</li>
<li>Generate the resource statement table</li>
</ol>

<div class="tip"><b>Tip:</b> Classification boundaries should be reviewed by a
Competent Person. GeoX provides the quantitative framework, but professional
judgement is essential for JORC/SAMREC compliance.</div>
""")

        c["resources/reporting"] = ("Resource Reporting", """
<h2>Resource Reporting</h2>
<p>Generate JORC/SAMREC-compliant resource tables summarising tonnage and grade
by classification category. Access via <b>Resources &gt; Resource Reporting</b>.</p>

<h3>Report Contents</h3>
<ul>
<li><b>Tonnage by category</b> &mdash; Measured, Indicated, Inferred, and Total</li>
<li><b>Grade by category</b> &mdash; mass-weighted average grades</li>
<li><b>Metal content</b> &mdash; contained metal (tonnage &times; grade)</li>
<li><b>Domain breakdown</b> &mdash; optional reporting by geological domain</li>
<li><b>Cut-off sensitivity</b> &mdash; how tonnage and grade change with cut-off</li>
</ul>

<h3>Export</h3>
<p>Reports can be exported as formatted tables suitable for inclusion in
technical reports and public disclosures.</p>
""")

        c["resources/grade_tonnage"] = ("Grade-Tonnage Curves", """
<h2>Grade-Tonnage Curves</h2>
<p>Grade-tonnage analysis shows how tonnes, grade, and metal content change
as the cut-off grade varies. GeoX provides three levels of analysis.</p>

<h3>Basic Grade-Tonnage</h3>
<p>Access via <b>Resources &gt; Basic Grade-Tonnage</b>. Simple curves without
economic optimization. Shows tonnage above cut-off, average grade above cut-off,
and metal content above cut-off.</p>

<h3>Full Grade-Tonnage &amp; Cut-off</h3>
<p>Access via <b>Resources &gt; Grade-Tonnage &amp; Cut-off</b>. Adds economic
parameters (metal price, mining cost, processing cost, recovery) to compute
profit and identify the optimal cut-off grade.</p>

<h3>Cut-off Optimization</h3>
<p>Access via <b>Resources &gt; Cut-off Optimization</b>. Full economic
optimization with NPV, IRR, and sensitivity analysis for metal price, cost,
and recovery scenarios.</p>
""")

        c["resources/block_calc"] = ("Block Property Calculator", """
<h2>Block Property Calculator</h2>
<p>Add derived properties to your block model such as tonnage, volume, and
metal content. Access via <b>Resources &gt; Block Property Calculator</b>.</p>

<h3>Available Calculations</h3>
<ul>
<li><b>Block volume</b> &mdash; computed from block dimensions</li>
<li><b>Block tonnage</b> &mdash; volume &times; density</li>
<li><b>Metal content</b> &mdash; tonnage &times; grade / conversion factor</li>
<li><b>Custom formula</b> &mdash; user-defined mathematical expressions using
existing block properties</li>
</ul>
""")

        # ==============================================================
        # MINE PLANNING
        # ==============================================================

        c["planning/pit"] = ("Pit Optimization", """
<h2>Pit Optimization</h2>
<p>GeoX implements the Lerchs-Grossmann algorithm for determining the ultimate
pit limit that maximises undiscounted profit. Access via
<b>Planning &gt; Pit Optimisation</b>.</p>

<h3>Input Parameters</h3>
<table>
<tr><th>Parameter</th><th>Description</th></tr>
<tr><td>Block model</td><td>Estimated grade model with tonnage/density</td></tr>
<tr><td>Metal price</td><td>Revenue per unit of metal</td></tr>
<tr><td>Mining cost</td><td>Cost per tonne of material mined</td></tr>
<tr><td>Processing cost</td><td>Cost per tonne of ore processed</td></tr>
<tr><td>Recovery</td><td>Metallurgical recovery factor (0&ndash;1)</td></tr>
<tr><td>Slope angles</td><td>Overall pit slope angles by sector</td></tr>
</table>

<h3>Output</h3>
<ul>
<li><b>Ultimate pit shell</b> &mdash; the pit boundary that maximises profit</li>
<li><b>Nested pit shells</b> &mdash; series of pits at different revenue factors
(used for pushback design)</li>
<li><b>Pit statistics</b> &mdash; tonnage, grade, strip ratio, profit</li>
</ul>
""")

        c["planning/pushback"] = ("Pushback Design", """
<h2>Pushback Design</h2>
<p>Pushback design sequences the extraction of the ultimate pit into practical
mining phases. Access via <b>Planning &gt; Pushback Design</b>.</p>

<h3>Approach</h3>
<p>GeoX uses nested pit shells (from pit optimization at different revenue
factors) to define pushback boundaries. Each pushback represents a phase
of mining that maintains slope angle requirements.</p>

<h3>Parameters</h3>
<ul>
<li><b>Number of pushbacks</b> &mdash; target number of mining phases</li>
<li><b>Revenue factors</b> &mdash; range of factors to generate nested shells</li>
<li><b>Minimum mining width</b> &mdash; practical constraint for equipment access</li>
</ul>
""")

        c["planning/bench"] = ("Bench Design", """
<h2>Bench Design</h2>
<p>Define bench geometry for open pit mine design. Access via
<b>Planning &gt; Bench Design</b>.</p>

<h3>Parameters</h3>
<ul>
<li><b>Bench height</b> &mdash; vertical distance between bench floors</li>
<li><b>Berm width</b> &mdash; horizontal safety berm between benches</li>
<li><b>Face angle</b> &mdash; angle of the bench face</li>
<li><b>Ramp width</b> &mdash; haul road width for truck access</li>
<li><b>Ramp gradient</b> &mdash; maximum slope for haul roads</li>
</ul>
""")

        c["planning/strategic"] = ("Strategic Scheduling", """
<h2>Strategic Scheduling</h2>
<p>Long-term (Life of Mine) production planning using mathematical optimization.
Access via <b>Planning &gt; Strategic Schedule</b>.</p>

<h3>Method</h3>
<p>GeoX uses Mixed Integer Linear Programming (MILP) with nested pit shells
and cut-off grade optimization to maximise NPV over the mine life.</p>

<h3>Inputs</h3>
<ul>
<li>Pushback design (mining phases)</li>
<li>Processing plant capacity (tonnes per year)</li>
<li>Mining capacity (tonnes per year)</li>
<li>Economic parameters (prices, costs, discount rate)</li>
<li>Grade blending constraints</li>
</ul>

<h3>Output</h3>
<ul>
<li>Annual production schedule (which blocks to mine each year)</li>
<li>NPV profile over mine life</li>
<li>Cash flow projections</li>
</ul>
""")

        c["planning/tactical"] = ("Tactical Scheduling", """
<h2>Tactical Scheduling</h2>
<p>Medium-term scheduling (monthly/quarterly) that translates the strategic
plan into operational targets. Access via <b>Planning &gt; Tactical Schedule</b>.</p>

<h3>Features</h3>
<ul>
<li>Monthly production targets by pushback</li>
<li>Bench sequencing within pushbacks</li>
<li>Development scheduling (ramps, access roads)</li>
<li>Equipment allocation per period</li>
</ul>
""")

        c["planning/short_term"] = ("Short-Term Scheduling", """
<h2>Short-Term Scheduling</h2>
<p>Weekly and daily scheduling for operational execution. Access via
<b>Planning &gt; Short-Term Schedule</b>.</p>

<h3>Features</h3>
<ul>
<li>Shift-level planning (day/night shifts)</li>
<li>Digline positioning for excavators</li>
<li>Truck dispatch optimization</li>
<li>Grade control integration</li>
<li>Stockpile management</li>
</ul>
""")

        c["planning/underground"] = ("Underground Mining", """
<h2>Underground Mining</h2>
<p>GeoX includes tools for underground mine design and analysis. Access via
<b>Planning &gt; Underground Mining</b>.</p>

<h3>Features</h3>
<ul>
<li><b>Stope design</b> &mdash; optimise stope dimensions for maximum value</li>
<li><b>Stope stability</b> &mdash; Mathews stability graph method for open stope design</li>
<li><b>Rockburst assessment</b> &mdash; seismic hazard evaluation</li>
<li><b>Slope stability</b> &mdash; pillar and crown pillar analysis</li>
</ul>
""")

        c["planning/fleet"] = ("Fleet Management", """
<h2>Fleet Management</h2>
<p>Configure and optimise mining fleet operations. Access via
<b>Planning &gt; Fleet Management</b>.</p>

<h3>Features</h3>
<ul>
<li>Equipment specification (truck capacity, speed, fuel consumption)</li>
<li>Cycle time analysis (load, haul, dump, return)</li>
<li>Fleet sizing for production targets</li>
<li>Dispatch optimization</li>
<li>Operating cost estimation</li>
</ul>
""")

        c["planning/economics"] = ("NPV / IRR Economics", """
<h2>NPV / IRR Economics</h2>
<p>Financial analysis tools for mine valuation and investment decisions.</p>

<h3>NPV Scheduling (NPVS)</h3>
<p>Access via <b>Planning &gt; NPVS Optimisation</b>. Optimises the production
schedule to maximise Net Present Value by considering the time value of money.</p>

<h3>IRR Analysis</h3>
<p>Access via <b>Planning &gt; IRR Analysis</b>. Computes the Internal Rate of
Return and risk-adjusted financial metrics.</p>

<h3>Parameters</h3>
<ul>
<li><b>Discount rate</b> &mdash; cost of capital for NPV calculation</li>
<li><b>Metal price assumptions</b> &mdash; base case and sensitivity scenarios</li>
<li><b>Capital expenditure</b> &mdash; upfront and sustaining capital</li>
<li><b>Operating expenditure</b> &mdash; mining, processing, G&amp;A costs</li>
<li><b>Tax and royalties</b> &mdash; fiscal regime parameters</li>
</ul>
""")

        # ==============================================================
        # VISUALIZATION
        # ==============================================================

        c["vis/viewer"] = ("3D Viewer Controls", """
<h2>3D Viewer Controls</h2>
<p>The GeoX 3D viewer is powered by VTK/PyVista and provides interactive
visualization of block models, drillholes, surfaces, and geological models.</p>

<h3>Mouse Controls</h3>
<table>
<tr><th>Action</th><th>Control</th></tr>
<tr><td>Orbit / Rotate</td><td>Left mouse button drag</td></tr>
<tr><td>Pan</td><td>Middle mouse button drag (or Shift + Left drag)</td></tr>
<tr><td>Zoom</td><td>Scroll wheel</td></tr>
<tr><td>Pick / Select</td><td>Left click (in Select mode)</td></tr>
</table>

<h3>Quick View Presets</h3>
<table>
<tr><th>Key</th><th>View</th></tr>
<tr><td><kbd>1</kbd></td><td>Top (plan view)</td></tr>
<tr><td><kbd>2</kbd></td><td>Bottom</td></tr>
<tr><td><kbd>3</kbd></td><td>Front (looking North)</td></tr>
<tr><td><kbd>4</kbd></td><td>Back (looking South)</td></tr>
<tr><td><kbd>5</kbd></td><td>Right (looking West)</td></tr>
<tr><td><kbd>6</kbd></td><td>Left (looking East)</td></tr>
<tr><td><kbd>7</kbd></td><td>Isometric (3D perspective)</td></tr>
</table>

<h3>Other Controls</h3>
<ul>
<li><kbd>R</kbd> &mdash; Reset camera to fit all visible data</li>
<li><kbd>F</kbd> &mdash; Fit view to current selection</li>
<li><kbd>O</kbd> &mdash; Toggle orthographic/perspective projection</li>
<li><kbd>+</kbd> / <kbd>-</kbd> &mdash; Zoom in / out</li>
</ul>
""")

        c["vis/display"] = ("Display Settings", """
<h2>Display Settings</h2>
<p>Control how data is rendered in the 3D viewer. Access via the
<b>Display Settings</b> panel or <kbd>Ctrl+D</kbd>.</p>

<h3>Rendering Options</h3>
<ul>
<li><b>Colormap</b> &mdash; select from scientific colormaps (viridis, turbo, plasma, etc.)</li>
<li><b>Color range</b> &mdash; set min/max values for the colormap</li>
<li><b>Opacity</b> &mdash; adjust transparency of blocks</li>
<li><b>Block edges</b> &mdash; toggle edge wireframe visibility (disable for cleaner
rendering of dense models)</li>
<li><b>Background colour</b> &mdash; light or dark background</li>
</ul>

<h3>Active Property</h3>
<p>Select which property to display by clicking on its name in the property
list. The colormap and legend update automatically.</p>

<div class="tip"><b>Tip:</b> For block models with more than 50,000 cells, edges
are automatically disabled to prevent GPU timeouts. You can re-enable them
from the Display Settings panel.</div>
""")

        c["vis/camera"] = ("Camera & Navigation", """
<h2>Camera &amp; Navigation</h2>

<h3>Camera Presets</h3>
<p>Use number keys <kbd>1</kbd>&ndash;<kbd>7</kbd> for instant camera presets
(Top, Bottom, Front, Back, Right, Left, Isometric).</p>

<h3>View Bookmarks</h3>
<p>Save and recall custom viewpoints:</p>
<ul>
<li><kbd>Ctrl+Shift+1</kbd> through <kbd>Ctrl+Shift+9</kbd> &mdash; save current view to slot</li>
<li><kbd>Ctrl+Shift+F1</kbd> through <kbd>Ctrl+Shift+F9</kbd> &mdash; load saved view from slot</li>
</ul>

<h3>Fine Navigation</h3>
<ul>
<li><b>Arrow keys</b> &mdash; nudge (pan) the view in small increments</li>
<li><b>Ctrl + Arrow keys</b> &mdash; rotate the view by 15&deg; increments</li>
<li><kbd>Home</kbd> &mdash; reset view to default</li>
</ul>

<h3>Projection Mode</h3>
<p>Press <kbd>O</kbd> to toggle between:</p>
<ul>
<li><b>Perspective</b> &mdash; realistic depth perception (default)</li>
<li><b>Orthographic</b> &mdash; no foreshortening, better for measurements and sections</li>
</ul>
""")

        c["vis/mouse"] = ("Mouse Modes", """
<h2>Mouse Modes</h2>
<p>GeoX supports multiple mouse interaction modes. Switch between them using
keyboard shortcuts or the Mouse panel.</p>

<h3>Available Modes</h3>
<table>
<tr><th>Mode</th><th>Shortcut</th><th>Description</th></tr>
<tr><td><b>Select</b></td><td><kbd>S</kbd></td><td>Click to pick/select blocks or drillholes</td></tr>
<tr><td><b>Pan</b></td><td><kbd>P</kbd></td><td>Drag to pan the view</td></tr>
<tr><td><b>Zoom Box</b></td><td><kbd>Z</kbd></td><td>Draw a rectangle to zoom into</td></tr>
<tr><td><b>Toggle</b></td><td><kbd>Space</kbd></td><td>Toggle between Select and Pan modes</td></tr>
</table>
""")

        c["vis/clip"] = ("Clip Planes", """
<h2>Clip Planes</h2>
<p>Clip planes allow you to slice through your 3D data to view internal
structure. Access via <b>Tools &gt; Clip Plane</b> or <kbd>Ctrl+Shift+X</kbd>.</p>

<h3>Features</h3>
<ul>
<li><b>Interactive positioning</b> &mdash; drag the clip plane in the 3D viewer</li>
<li><b>Normal direction</b> &mdash; orient the clip plane along X, Y, Z, or custom direction</li>
<li><b>Offset</b> &mdash; slide the plane position along its normal</li>
<li><b>Invert</b> &mdash; flip which side is clipped</li>
</ul>

<div class="tip"><b>Tip:</b> Clip planes are excellent for inspecting kriging
or simulation results at specific bench elevations.</div>
""")

        c["vis/legend"] = ("Legends & Scale Bar", """
<h2>Legends &amp; Scale Bar</h2>

<h3>Legend</h3>
<p>The legend automatically displays the active property name, colormap, and
value range. It can be customised via the legend widget controls:</p>
<ul>
<li><b>Position</b> &mdash; drag to reposition anywhere in the viewer</li>
<li><b>Size</b> &mdash; resize for readability</li>
<li><b>Label format</b> &mdash; number of decimal places, scientific notation</li>
</ul>

<h3>Scale Bar</h3>
<p>An optional scale bar shows real-world distances. Configure via
<b>View &gt; Axes &amp; Scale Bar</b>.</p>

<h3>North Arrow</h3>
<p>An optional north arrow indicates geographic orientation. Useful for
plan-view screenshots and reports.</p>
""")

        c["vis/screenshot"] = ("Screenshot Export", """
<h2>Screenshot Export</h2>
<p>Export high-resolution screenshots for reports and presentations. Access via
<b>File &gt; Advanced Screenshot</b> or <kbd>Ctrl+Shift+S</kbd>.</p>

<h3>Options</h3>
<ul>
<li><b>Resolution</b> &mdash; set custom width and height (up to 8K)</li>
<li><b>Background</b> &mdash; white, black, transparent, or current</li>
<li><b>Include legend</b> &mdash; toggle legend visibility in export</li>
<li><b>Include scale bar</b> &mdash; toggle scale bar visibility</li>
<li><b>Format</b> &mdash; PNG, JPEG, TIFF, or SVG</li>
</ul>
""")

        c["vis/layout"] = ("Layout Export", """
<h2>Layout Export</h2>
<p>Create multi-panel layouts for professional reports. Access via
<b>File &gt; Layout Export</b>.</p>

<h3>Features</h3>
<ul>
<li>Multiple viewport arrangement (plan view, sections, 3D perspective)</li>
<li>Title block with project information</li>
<li>Legend synchronised with main viewer</li>
<li>Scale bar and north arrow</li>
<li>Export to PDF or PNG</li>
</ul>

<div class="tip"><b>Tip:</b> Layout legends automatically sync with the main viewer.
Set up your preferred property and colormap before exporting.</div>
""")

        # ==============================================================
        # TOOLS & UTILITIES
        # ==============================================================

        c["tools/palette"] = ("Command Palette", """
<h2>Command Palette</h2>
<p>The Command Palette provides instant access to every feature in GeoX.
Press <kbd>Ctrl+Shift+P</kbd> to open it.</p>

<h3>Usage</h3>
<ul>
<li>Start typing to fuzzy-search across all panels, tools, and actions</li>
<li>Press <kbd>Enter</kbd> to activate the selected command</li>
<li>Press <kbd>Esc</kbd> to close</li>
</ul>

<h3>Search Scope</h3>
<p>The palette searches across:</p>
<ul>
<li>All registered panels (70+)</li>
<li>Menu actions (File, Edit, View, etc.)</li>
<li>Keyboard shortcuts</li>
<li>Data registry entries</li>
</ul>
""")

        c["tools/registry"] = ("Data Registry", """
<h2>Data Registry</h2>
<p>The Data Registry is the central data management system in GeoX. It tracks
all loaded datasets, their processing history, and relationships.</p>

<h3>Viewing the Registry</h3>
<p>Access via <b>Tools &gt; Data Registry Status</b> (requires Developer Mode).
This shows all registered datasets, their types, and lineage.</p>

<h3>Key Principles</h3>
<ul>
<li><b>Immutability</b> &mdash; raw data is never modified; transformations create
new entries</li>
<li><b>Provenance</b> &mdash; every dataset records its parent sources and the
operation that created it</li>
<li><b>Authoritative source</b> &mdash; the block model is the single source of
truth for spatial properties</li>
</ul>
""")

        c["tools/statistics"] = ("Statistics Panel", """
<h2>Statistics Panel</h2>
<p>Compute descriptive statistics for any numeric property. Access via
<b>Tools &gt; Statistics</b>.</p>

<h3>Available Statistics</h3>
<ul>
<li>Count, mean, median, mode</li>
<li>Standard deviation, variance, coefficient of variation</li>
<li>Minimum, maximum, range</li>
<li>Percentiles (P10, P25, P50, P75, P90)</li>
<li>Skewness and kurtosis</li>
</ul>

<h3>Visualizations</h3>
<ul>
<li><b>Histogram</b> &mdash; frequency distribution</li>
<li><b>Cumulative distribution</b> &mdash; CDF plot</li>
<li><b>Box plot</b> &mdash; quartile summary</li>
<li><b>Q-Q plot</b> &mdash; normality assessment</li>
</ul>
""")

        c["tools/charts"] = ("Charts Panel", """
<h2>Charts Panel</h2>
<p>Create customisable charts from your data. Access via <b>Tools &gt; Charts</b>.</p>

<h3>Chart Types</h3>
<ul>
<li>Scatter plots (2D and 3D)</li>
<li>Line charts</li>
<li>Bar charts</li>
<li>Histograms</li>
<li>Correlation matrices</li>
</ul>

<h3>Features</h3>
<ul>
<li>Select any property columns for X, Y, Z axes</li>
<li>Color by a third variable</li>
<li>Export charts as images (PNG, SVG)</li>
<li>Interactive zoom and pan</li>
</ul>
""")

        c["tools/export"] = ("Data Export", """
<h2>Data Export</h2>
<p>Export your data in various formats. Access via <b>File &gt; Export</b>.</p>

<h3>Export Formats</h3>
<table>
<tr><th>Format</th><th>Extension</th><th>Use Case</th></tr>
<tr><td>CSV</td><td>.csv</td><td>Tabular data for spreadsheets and other software</td></tr>
<tr><td>Excel</td><td>.xlsx</td><td>Formatted spreadsheets with multiple sheets</td></tr>
<tr><td>VTK</td><td>.vtk</td><td>3D grid data for visualization tools</td></tr>
<tr><td>STL</td><td>.stl</td><td>Surface meshes for CAD software</td></tr>
<tr><td>OBJ</td><td>.obj</td><td>3D models for general 3D tools</td></tr>
</table>
""")

        c["tools/preferences"] = ("Preferences", """
<h2>Preferences</h2>
<p>Configure application settings. Access via <b>Edit &gt; Preferences</b>
or <kbd>Ctrl+,</kbd>.</p>

<h3>Settings</h3>
<ul>
<li><b>Theme</b> &mdash; Dark or Light mode</li>
<li><b>Default colormap</b> &mdash; choose the default colormap for new visualizations</li>
<li><b>Units</b> &mdash; metric or imperial</li>
<li><b>Performance</b> &mdash; level-of-detail thresholds, GPU settings</li>
<li><b>Auto-save</b> &mdash; enable/disable automatic project saving</li>
</ul>
""")

        # ==============================================================
        # KEYBOARD SHORTCUTS
        # ==============================================================

        c["shortcuts/file"] = ("File Operations", """
<h2>File Operations Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+O</kbd></td><td>Open File</td></tr>
<tr><td><kbd>Ctrl+N</kbd></td><td>New Project</td></tr>
<tr><td><kbd>Ctrl+S</kbd></td><td>Save Project</td></tr>
<tr><td><kbd>Ctrl+Shift+S</kbd></td><td>Export Screenshot</td></tr>
<tr><td><kbd>Ctrl+1</kbd> ... <kbd>Ctrl+9</kbd></td><td>Open Recent File 1&ndash;9</td></tr>
<tr><td><kbd>Ctrl+W</kbd></td><td>Clear Scene</td></tr>
<tr><td><kbd>Ctrl+D</kbd></td><td>View Block Model Data</td></tr>
<tr><td><kbd>Ctrl+Q</kbd></td><td>Exit Application</td></tr>
</table>
""")

        c["shortcuts/view"] = ("View Controls", """
<h2>View Control Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>R</kbd></td><td>Reset View</td></tr>
<tr><td><kbd>F</kbd></td><td>Fit to Model</td></tr>
<tr><td><kbd>E</kbd></td><td>Zoom to Extents</td></tr>
<tr><td><kbd>Home</kbd></td><td>Reset View to Default</td></tr>
<tr><td><kbd>+</kbd> / <kbd>=</kbd></td><td>Zoom In</td></tr>
<tr><td><kbd>-</kbd></td><td>Zoom Out</td></tr>
<tr><td><kbd>F5</kbd></td><td>Refresh View</td></tr>
</table>
""")

        c["shortcuts/camera"] = ("Camera Navigation", """
<h2>Camera Navigation Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Arrow Keys</kbd></td><td>Nudge View (Pan)</td></tr>
<tr><td><kbd>Ctrl+Arrow Keys</kbd></td><td>Rotate View 15&deg;</td></tr>
<tr><td><kbd>1</kbd></td><td>Top View</td></tr>
<tr><td><kbd>2</kbd></td><td>Bottom View</td></tr>
<tr><td><kbd>3</kbd></td><td>Front View</td></tr>
<tr><td><kbd>4</kbd></td><td>Back View</td></tr>
<tr><td><kbd>5</kbd></td><td>Right View</td></tr>
<tr><td><kbd>6</kbd></td><td>Left View</td></tr>
<tr><td><kbd>7</kbd></td><td>Isometric View</td></tr>
</table>
""")

        c["shortcuts/bookmarks"] = ("View Bookmarks", """
<h2>View Bookmark Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+Shift+1</kbd> ... <kbd>Ctrl+Shift+9</kbd></td><td>Save Current View to Bookmark Slot 1&ndash;9</td></tr>
<tr><td><kbd>Ctrl+Shift+F1</kbd> ... <kbd>Ctrl+Shift+F9</kbd></td><td>Load Saved View from Bookmark Slot 1&ndash;9</td></tr>
</table>

<div class="tip"><b>Tip:</b> View bookmarks are saved with your project, so you can
return to important viewpoints across sessions.</div>
""")

        c["shortcuts/display"] = ("Display Toggles", """
<h2>Display Toggle Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>O</kbd></td><td>Toggle Orthographic / Perspective Projection</td></tr>
<tr><td><kbd>A</kbd></td><td>Toggle Axes Visibility</td></tr>
<tr><td><kbd>G</kbd></td><td>Toggle Grid Visibility</td></tr>
</table>
""")

        c["shortcuts/mouse"] = ("Mouse Modes", """
<h2>Mouse Mode Shortcuts</h2>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>S</kbd></td><td>Select (Click) Mode</td></tr>
<tr><td><kbd>P</kbd></td><td>Pan Mode</td></tr>
<tr><td><kbd>Z</kbd></td><td>Zoom Box Mode</td></tr>
<tr><td><kbd>Space</kbd></td><td>Toggle Select / Pan Mode</td></tr>
<tr><td><kbd>F1</kbd></td><td>Show Documentation (this window)</td></tr>
<tr><td><kbd>Ctrl+/</kbd></td><td>Show Keyboard Shortcuts Dialog</td></tr>
<tr><td><kbd>Ctrl+Shift+P</kbd></td><td>Open Command Palette</td></tr>
</table>
""")
