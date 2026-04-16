"""
Report Generator — Word/PDF Resource Reports
==============================================

Generates professional Word (.docx) resource reports suitable for
feasibility-study appendices. Includes:

- Block 8: Full resource statement report with tables, GT curves, audit trail
- Block 9: JORC Table 1 Section 3 (2-column Criteria / Commentary)

Requires ``python-docx``.  Falls back gracefully if not installed.

Author: GeoX Mining Software Platform
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  python-docx availability check
# --------------------------------------------------------------------------- #
try:
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor, Emu
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn, nsdecls
    from docx.oxml import parse_xml
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# --------------------------------------------------------------------------- #
#  Standard Disclaimer Templates
# --------------------------------------------------------------------------- #

_DISCLAIMERS = {
    "JORC 2012": (
        "The information in this report that relates to Mineral Resources "
        "is based on information compiled by {cp_name}, a Competent Person "
        "who is a Member of {membership}. {cp_name} has sufficient experience "
        "that is relevant to the style of mineralisation and type of deposit "
        "under consideration and to the activity being undertaken to qualify "
        "as a Competent Person as defined in the 2012 Edition of the JORC Code. "
        "{cp_name} consents to the inclusion in the report of the matters "
        "based on his/her information in the form and context in which it appears."
    ),
    "SAMREC 2016": (
        "The Mineral Resource has been estimated and classified in accordance "
        "with the South African Code for Reporting of Exploration Results, "
        "Mineral Resources and Mineral Reserves (SAMREC Code, 2016 Edition)."
    ),
    "NI 43-101 / CIM": (
        "The Mineral Resource estimate has been prepared in accordance with "
        "the Canadian Institute of Mining, Metallurgy and Petroleum Definition "
        "Standards incorporated by reference in NI 43-101."
    ),
}


def get_disclaimer(reporting_code: str, cp_info: dict) -> str:
    """Return the formatted disclaimer string for *reporting_code*."""
    template = _DISCLAIMERS.get(reporting_code, "")
    if not template:
        return ""
    return template.format(
        cp_name=cp_info.get("name", "[Competent Person]"),
        membership=cp_info.get("membership", "[Professional Membership]"),
    )


# --------------------------------------------------------------------------- #
#  Helper: table cell shading
# --------------------------------------------------------------------------- #

def _shade_cell(cell, hex_color: str = "4472C4"):
    """Apply background shading to a table cell."""
    shading = parse_xml(
        f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>'
    )
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_text(cell, text: str, bold: bool = False, align=None,
                   font_size: int = 10, font_color: RGBColor | None = None):
    """Set cell text with formatting."""
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    run.font.size = Pt(font_size)
    run.bold = bold
    if font_color:
        run.font.color.rgb = font_color


def _add_table_borders(table):
    """Add visible borders to all cells in a Word table."""
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(
        f'<w:tblPr {nsdecls("w")}/>'
    )
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '  <w:left w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '  <w:right w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)


# --------------------------------------------------------------------------- #
#  Helper: build a resource table from a ResourceSummaryResult
# --------------------------------------------------------------------------- #

def _add_resource_table(doc, result, grade_field: str = "Grade",
                        title: str | None = None):
    """Insert a formatted resource table into *doc*."""
    if title:
        doc.add_heading(title, level=3)

    headers = [
        "Classification", "Blocks", "Volume (m\u00b3)",
        "Density (t/m\u00b3)", "Tonnes (t)",
        f"{grade_field} (%)", "Contained Metal (t)",
    ]

    all_rows = list(result.rows)
    if result.totals_MI:
        all_rows.append(result.totals_MI)
    if result.totals_all:
        all_rows.append(result.totals_all)

    table = doc.add_table(rows=1 + len(all_rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    _add_table_borders(table)

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        _set_cell_text(cell, h, bold=True, font_size=9,
                       font_color=RGBColor(0xFF, 0xFF, 0xFF),
                       align=WD_ALIGN_PARAGRAPH.CENTER)
        _shade_cell(cell, "4472C4")

    # Data rows
    for r_idx, row in enumerate(all_rows, start=1):
        is_total = row.classification.startswith(("Totals", "COMBINED",
                                                   "Measured + Indicated"))
        values = [
            row.classification,
            f"{row.n_blocks:,}",
            f"{row.total_volume_m3:,.0f}",
            f"{row.avg_density_t_per_m3:.2f}",
            f"{row.total_tonnage_t:,.0f}",
            f"{row.grade_pct:.2f}",
            f"{row.contained_metal_t:,.0f}",
        ]
        for c_idx, val in enumerate(values):
            cell = table.rows[r_idx].cells[c_idx]
            align = (WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0
                     else WD_ALIGN_PARAGRAPH.RIGHT)
            _set_cell_text(cell, val, bold=is_total, font_size=9, align=align)
            if is_total:
                _shade_cell(cell, "D9E2F3")

    doc.add_paragraph("")  # spacing


# =========================================================================== #
#  Block 8 — Full Resource Report (.docx)
# =========================================================================== #

def generate_resource_report(
    summary_result,
    domain_results: list | None,
    config: dict,
    cp_info: dict,
    reporting_code: str,
    gt_image_path: str | None,
    output_path: str,
) -> str:
    """Generate a professional Word (.docx) resource report.

    Parameters
    ----------
    summary_result : ResourceSummaryResult
        The combined resource summary from the engine.
    domain_results : list | None
        List of ``(domain_name, ResourceSummaryResult)`` tuples, or *None*.
    config : dict
        Panel configuration — grade_field, class_field, density_mode, etc.
    cp_info : dict
        Competent person information — name, quals, membership, effective_date.
    reporting_code : str
        One of ``"JORC 2012"``, ``"SAMREC 2016"``, ``"NI 43-101 / CIM"``,
        or ``"None"``.
    gt_image_path : str | None
        Path to a saved grade-tonnage plot PNG, or *None*.
    output_path : str
        Destination ``.docx`` path.

    Returns
    -------
    str
        The absolute path to the generated file.

    Raises
    ------
    ImportError
        If ``python-docx`` is not installed.
    """
    if not DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is required for Word report generation.\n"
            "Install it with:  pip install python-docx"
        )

    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    grade_field = config.get("grade_field", "Grade")
    project_name = config.get("project_name", "Mineral Resource Estimate")
    effective_date = cp_info.get("effective_date", datetime.now().strftime("%d %B %Y"))

    # ------------------------------------------------------------------ #
    #  Cover Page
    # ------------------------------------------------------------------ #
    for _ in range(6):
        doc.add_paragraph("")

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run("MINERAL RESOURCE STATEMENT")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    doc.add_paragraph("")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(project_name)
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x34, 0x49, 0x5E)

    doc.add_paragraph("")

    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run(f"Effective Date: {effective_date}")
    run.font.size = Pt(12)

    doc.add_paragraph("")

    # DRAFT watermark note
    draft_p = doc.add_paragraph()
    draft_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = draft_p.add_run("DRAFT - FOR REVIEW PURPOSES ONLY")
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0xE7, 0x4C, 0x3C)

    if cp_info.get("name"):
        doc.add_paragraph("")
        cp_p = doc.add_paragraph()
        cp_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp_p.add_run(f"Prepared by: {cp_info['name']}")
        run.font.size = Pt(11)
        if cp_info.get("quals"):
            doc.add_paragraph("")
            q_p = doc.add_paragraph()
            q_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = q_p.add_run(cp_info["quals"])
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)

    doc.add_page_break()

    # ------------------------------------------------------------------ #
    #  Section 1 — Resource Statement
    # ------------------------------------------------------------------ #
    doc.add_heading("1. Resource Statement", level=1)

    cutoff_grade = config.get("cutoff_grade")
    density_mode = config.get("density_mode", "constant")
    density_value = config.get("density_value")

    intro_parts = [
        f"The Mineral Resource has been estimated and classified for the "
        f"{project_name} project."
    ]
    if cutoff_grade is not None:
        intro_parts.append(
            f"Resources are reported above a cutoff grade of "
            f"{cutoff_grade} {grade_field}."
        )
    if density_mode == "constant" and density_value:
        intro_parts.append(
            f"A constant bulk density of {density_value} t/m\u00b3 has been applied."
        )
    elif density_mode == "domain":
        intro_parts.append(
            "Domain-specific bulk densities have been applied."
        )
    elif density_mode == "block":
        intro_parts.append(
            "Per-block density values from the block model have been used."
        )

    doc.add_paragraph(" ".join(intro_parts))

    # Combined resource table
    _add_resource_table(doc, summary_result, grade_field,
                        title="Combined Resource Summary")

    # Per-domain tables
    if domain_results:
        doc.add_heading("Domain-Level Resource Summaries", level=2)
        for domain_name, domain_result in domain_results:
            _add_resource_table(doc, domain_result, grade_field,
                                title=f"Domain: {domain_name}")

    # Disclaimer
    disclaimer = get_disclaimer(reporting_code, cp_info)
    if disclaimer:
        doc.add_paragraph("")
        disc_p = doc.add_paragraph()
        disc_p.style = doc.styles["Intense Quote"] if "Intense Quote" in [
            s.name for s in doc.styles
        ] else doc.styles["Normal"]
        run = disc_p.add_run(disclaimer)
        run.font.size = Pt(9)
        run.italic = True

    doc.add_page_break()

    # ------------------------------------------------------------------ #
    #  Section 2 — Grade-Tonnage Curves
    # ------------------------------------------------------------------ #
    doc.add_heading("2. Grade-Tonnage Curves", level=1)

    if gt_image_path and Path(gt_image_path).exists():
        doc.add_paragraph(
            "The following grade-tonnage curves illustrate the sensitivity "
            "of the resource estimate to changes in cutoff grade."
        )
        try:
            doc.add_picture(gt_image_path, width=Inches(6.0))
            last_paragraph = doc.paragraphs[-1]
            last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        except Exception as exc:
            logger.warning("Could not embed GT image: %s", exc)
            doc.add_paragraph(f"[Image could not be embedded: {exc}]")
    else:
        doc.add_paragraph(
            "[Grade-tonnage plot not available. Generate a GT curve from "
            "the Resource Reporting panel and re-export.]"
        )

    doc.add_page_break()

    # ------------------------------------------------------------------ #
    #  Section 3 — Classification Summary
    # ------------------------------------------------------------------ #
    doc.add_heading("3. Classification Summary", level=1)

    doc.add_paragraph(
        "The resource has been classified according to the level of "
        "geological and grade confidence using the following categories:"
    )

    class_list = doc.add_paragraph()
    class_list.style = "List Bullet"
    class_list.add_run("Measured").bold = True
    class_list.add_run(
        " - Sufficient data density and quality to allow confident "
        "estimation of tonnage, grade, and geological continuity."
    )

    class_list2 = doc.add_paragraph()
    class_list2.style = "List Bullet"
    class_list2.add_run("Indicated").bold = True
    class_list2.add_run(
        " - Reasonable data density allowing estimation with a "
        "reasonable level of confidence."
    )

    class_list3 = doc.add_paragraph()
    class_list3.style = "List Bullet"
    class_list3.add_run("Inferred").bold = True
    class_list3.add_run(
        " - Limited data; geological and grade continuity assumed "
        "but not verified."
    )

    # Block counts per category
    doc.add_heading("Block Counts by Classification", level=2)
    cat_table = doc.add_table(rows=1, cols=3)
    cat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cat_table.style = "Table Grid"
    _add_table_borders(cat_table)

    for i, h in enumerate(["Classification", "Block Count", "% of Total"]):
        cell = cat_table.rows[0].cells[i]
        _set_cell_text(cell, h, bold=True, font_size=9,
                       font_color=RGBColor(0xFF, 0xFF, 0xFF),
                       align=WD_ALIGN_PARAGRAPH.CENTER)
        _shade_cell(cell, "4472C4")

    total_blocks = sum(r.n_blocks for r in summary_result.rows)
    for row in summary_result.rows:
        r = cat_table.add_row()
        pct = (row.n_blocks / total_blocks * 100) if total_blocks > 0 else 0
        _set_cell_text(r.cells[0], row.classification, font_size=9)
        _set_cell_text(r.cells[1], f"{row.n_blocks:,}", font_size=9,
                       align=WD_ALIGN_PARAGRAPH.RIGHT)
        _set_cell_text(r.cells[2], f"{pct:.1f}%", font_size=9,
                       align=WD_ALIGN_PARAGRAPH.RIGHT)

    doc.add_paragraph("")
    doc.add_paragraph(
        f"Classification field: {config.get('class_field', '[Not specified]')}"
    )

    doc.add_page_break()

    # ------------------------------------------------------------------ #
    #  Section 4 — Estimation Summary
    # ------------------------------------------------------------------ #
    doc.add_heading("4. Estimation Summary", level=1)

    est_items = [
        ("Estimation Method", config.get("estimation_method", "[Not available]")),
        ("Variogram Model", config.get("variogram_model", "[Not available]")),
        ("Search Radius", config.get("search_radius", "[Not available]")),
        ("Minimum Samples", config.get("min_samples", "[Not available]")),
        ("Maximum Samples", config.get("max_samples", "[Not available]")),
        ("Grade Field", grade_field),
        ("Density Mode", density_mode),
        ("Volume Mode", config.get("volume_mode", "[Not available]")),
    ]

    est_table = doc.add_table(rows=len(est_items), cols=2)
    est_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    est_table.style = "Table Grid"
    _add_table_borders(est_table)

    for i, (param, val) in enumerate(est_items):
        _set_cell_text(est_table.rows[i].cells[0], param, bold=True,
                       font_size=9)
        _set_cell_text(est_table.rows[i].cells[1], str(val), font_size=9)

    doc.add_page_break()

    # ------------------------------------------------------------------ #
    #  Section 5 — Audit Trail
    # ------------------------------------------------------------------ #
    doc.add_heading("5. Audit Trail", level=1)

    doc.add_paragraph(
        "The following information is provided for audit and reproducibility "
        "purposes."
    )

    try:
        from block_model_viewer import __version__ as app_version
    except ImportError:
        app_version = "Unknown"

    meta = summary_result.metadata if hasattr(summary_result, "metadata") else {}

    audit_items = [
        ("Software", f"GeoX Mining Software v{app_version}"),
        ("Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Effective Date", effective_date),
        ("Reporting Code", reporting_code if reporting_code != "None" else "Not specified"),
        ("Competent Person", cp_info.get("name", "[Not specified]")),
        ("Qualifications", cp_info.get("quals", "[Not specified]")),
        ("Professional Membership", cp_info.get("membership", "[Not specified]")),
        ("Execution Time", f"{meta.get('execution_time_seconds', 'N/A')} s"),
        ("Numba Accelerated", str(meta.get("numba_used", "N/A"))),
        ("Computation Timestamp", str(meta.get("timestamp", "N/A"))),
    ]

    audit_table = doc.add_table(rows=len(audit_items), cols=2)
    audit_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    audit_table.style = "Table Grid"
    _add_table_borders(audit_table)

    for i, (param, val) in enumerate(audit_items):
        _set_cell_text(audit_table.rows[i].cells[0], param, bold=True,
                       font_size=9)
        _set_cell_text(audit_table.rows[i].cells[1], str(val), font_size=9)

    # CP attestation at end
    if cp_info.get("name") and reporting_code != "None":
        doc.add_paragraph("")
        doc.add_paragraph("")
        sig_p = doc.add_paragraph()
        sig_p.add_run("_" * 40)
        doc.add_paragraph(f"{cp_info['name']}, {cp_info.get('quals', '')}")
        doc.add_paragraph(cp_info.get("membership", ""))
        doc.add_paragraph(f"Date: {effective_date}")

    # ------------------------------------------------------------------ #
    #  Save
    # ------------------------------------------------------------------ #
    output_path = str(output_path)
    if not output_path.lower().endswith(".docx"):
        output_path += ".docx"

    doc.save(output_path)
    logger.info("Resource report saved to %s", output_path)
    return output_path


# =========================================================================== #
#  Block 9 — JORC Table 1 Section 3
# =========================================================================== #

# The 14 criteria items for JORC Table 1 Section 3
_JORC_TABLE1_S3_ITEMS = [
    (
        "Database integrity",
        "Measures taken to ensure that data has not been corrupted by, for "
        "example, transcription or keying errors, between its initial "
        "collection and its use for Mineral Resource estimation purposes.",
    ),
    (
        "Site visits",
        "Comment on any site visits undertaken by the Competent Person and "
        "the outcome of those visits. If no site visits have been undertaken "
        "indicate why this is the case.",
    ),
    (
        "Geological interpretation",
        "Confidence in (or conversely, the uncertainty of) the geological "
        "interpretation of the mineral deposit. Nature of the data used and "
        "of any assumptions made.",
    ),
    (
        "Dimensions",
        "The extent and variability of the Mineral Resource expressed as "
        "length (along strike or otherwise), plan width, and depth below "
        "surface to the upper and lower limits of the Mineral Resource.",
    ),
    (
        "Estimation and modelling techniques",
        "The nature and appropriateness of the estimation technique(s) "
        "applied and key assumptions, including treatment of extreme grade "
        "values, domaining, interpolation parameters and maximum distance "
        "of extrapolation from data points.",
    ),
    (
        "Moisture",
        "Whether the tonnages are estimated on a dry basis or with natural "
        "moisture, and the method of determination of the moisture content.",
    ),
    (
        "Cut-off parameters",
        "The basis of the adopted cut-off grade(s) or quality parameters "
        "applied.",
    ),
    (
        "Mining factors or assumptions",
        "Assumptions made regarding possible mining methods, minimum mining "
        "dimensions and internal (or, if applicable, external) mining "
        "dilution.",
    ),
    (
        "Metallurgical factors or assumptions",
        "The basis for assumptions or predictions regarding metallurgical "
        "amenability.",
    ),
    (
        "Environmental factors or assumptions",
        "Assumptions made regarding possible waste and process residue "
        "disposal options.",
    ),
    (
        "Bulk density",
        "Whether assumed or determined. If assumed, the basis for the "
        "assumptions. If determined, the method used, whether wet or dry, "
        "the frequency of the measurements, the nature, size and "
        "representativeness of the samples.",
    ),
    (
        "Classification",
        "The basis for the classification of the Mineral Resources into "
        "varying confidence categories. Whether appropriate account has "
        "been taken of all relevant factors.",
    ),
    (
        "Audits or reviews",
        "The results of any audits or reviews of Mineral Resource estimates.",
    ),
    (
        "Discussion of relative accuracy/ confidence",
        "Where appropriate a statement of the relative accuracy and "
        "confidence level in the Mineral Resource estimate using an "
        "approach or procedure deemed appropriate by the Competent Person.",
    ),
]


def _auto_commentary(criteria: str, config: dict, cp_info: dict) -> str:
    """Return auto-populated commentary where config provides data."""
    placeholder = "[To be completed by Competent Person]"

    if criteria == "Estimation and modelling techniques":
        method = config.get("estimation_method")
        if method:
            return (
                f"Grade estimation was performed using {method}. "
                f"Variogram model: {config.get('variogram_model', '[Not available]')}. "
                f"Search radius: {config.get('search_radius', '[Not available]')}. "
                f"Min samples: {config.get('min_samples', '[Not available]')}. "
                f"Max samples: {config.get('max_samples', '[Not available]')}. "
                f"\n\n{placeholder}"
            )

    if criteria == "Cut-off parameters":
        cutoff = config.get("cutoff_grade")
        grade_field = config.get("grade_field", "grade")
        if cutoff is not None:
            return (
                f"A cutoff grade of {cutoff} {grade_field} has been applied. "
                f"\n\n{placeholder}"
            )

    if criteria == "Bulk density":
        mode = config.get("density_mode")
        if mode == "constant":
            val = config.get("density_value", "N/A")
            return (
                f"A constant bulk density of {val} t/m\u00b3 has been assumed. "
                f"\n\n{placeholder}"
            )
        elif mode == "domain":
            return (
                f"Domain-specific bulk densities have been applied. "
                f"\n\n{placeholder}"
            )
        elif mode == "block":
            return (
                f"Per-block density values from the block model column "
                f"'{config.get('density_field', '[column]')}' have been used. "
                f"\n\n{placeholder}"
            )

    if criteria == "Classification":
        class_field = config.get("class_field")
        if class_field:
            return (
                f"Classification is based on the '{class_field}' field "
                f"in the block model. "
                f"\n\n{placeholder}"
            )

    return placeholder


def generate_jorc_table1_section3(
    config: dict,
    cp_info: dict,
    output_path: str,
) -> str:
    """Generate a JORC Table 1 Section 3 document (.docx).

    Parameters
    ----------
    config : dict
        Panel configuration with available estimation parameters.
    cp_info : dict
        Competent person information.
    output_path : str
        Destination ``.docx`` path.

    Returns
    -------
    str
        The absolute path to the generated file.
    """
    if not DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is required for JORC Table 1 generation.\n"
            "Install it with:  pip install python-docx"
        )

    doc = Document()

    # Landscape for wide table
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Cm(29.7)
    section.page_height = Cm(21.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    # Title
    doc.add_heading("JORC Code, 2012 Edition - Table 1", level=1)
    doc.add_heading("Section 3: Estimation and Reporting of Mineral Resources", level=2)

    effective_date = cp_info.get("effective_date", datetime.now().strftime("%d %B %Y"))
    doc.add_paragraph(f"Effective Date: {effective_date}")
    if cp_info.get("name"):
        doc.add_paragraph(
            f"Competent Person: {cp_info['name']}, "
            f"{cp_info.get('quals', '')}, "
            f"{cp_info.get('membership', '')}"
        )

    doc.add_paragraph("")

    # 2-column table
    table = doc.add_table(rows=1 + len(_JORC_TABLE1_S3_ITEMS), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    _add_table_borders(table)

    # Set column widths (criteria ~30%, commentary ~70%)
    for row in table.rows:
        row.cells[0].width = Cm(7)
        row.cells[1].width = Cm(18)

    # Header
    for i, h in enumerate(["Criteria", "Commentary"]):
        cell = table.rows[0].cells[i]
        _set_cell_text(cell, h, bold=True, font_size=10,
                       font_color=RGBColor(0xFF, 0xFF, 0xFF),
                       align=WD_ALIGN_PARAGRAPH.CENTER)
        _shade_cell(cell, "4472C4")

    # Data rows
    for r_idx, (criteria, description) in enumerate(_JORC_TABLE1_S3_ITEMS, start=1):
        # Criteria column
        cell_criteria = table.rows[r_idx].cells[0]
        _set_cell_text(cell_criteria, criteria, bold=True, font_size=9)

        # Commentary column — auto-populate where possible
        commentary = _auto_commentary(criteria, config, cp_info)

        cell_commentary = table.rows[r_idx].cells[1]
        # Use description as guidance + auto commentary
        cell_commentary.text = ""
        p = cell_commentary.paragraphs[0]
        # Guidance text in grey italic
        guidance_run = p.add_run(f"({description})\n\n")
        guidance_run.font.size = Pt(8)
        guidance_run.italic = True
        guidance_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
        # Commentary
        commentary_run = p.add_run(commentary)
        commentary_run.font.size = Pt(9)

        # Alternate row shading
        if r_idx % 2 == 0:
            _shade_cell(cell_criteria, "F2F2F2")
            _shade_cell(cell_commentary, "F2F2F2")

    # Save
    output_path = str(output_path)
    if not output_path.lower().endswith(".docx"):
        output_path += ".docx"

    doc.save(output_path)
    logger.info("JORC Table 1 Section 3 saved to %s", output_path)
    return output_path
