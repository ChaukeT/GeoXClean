"""
Input validation for survey deformation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class ValidationIssue:
    table: str
    level: str  # ERROR or WARNING
    message: str


class SurveyDataValidator:
    """Schema-level validation for survey and groundwater inputs."""

    REQUIRED_SURVEY = ["point_id", "easting", "northing", "elevation", "survey_date"]
    REQUIRED_WELLS = ["well_id", "easting", "northing", "water_level", "date"]

    @staticmethod
    def validate_survey(df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        missing = [c for c in SurveyDataValidator.REQUIRED_SURVEY if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    table="subsidence_survey",
                    level="ERROR",
                    message=f"Missing required columns: {', '.join(missing)}",
                )
            )
            return issues

        if df["survey_date"].isna().any():
            issues.append(
                ValidationIssue(
                    table="subsidence_survey",
                    level="ERROR",
                    message="survey_date contains null/invalid entries",
                )
            )
        return issues

    @staticmethod
    def validate_wells(df: pd.DataFrame) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        missing = [c for c in SurveyDataValidator.REQUIRED_WELLS if c not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    table="groundwater",
                    level="ERROR",
                    message=f"Missing required columns: {', '.join(missing)}",
                )
            )
            return issues

        if df["date"].isna().any():
            issues.append(
                ValidationIssue(
                    table="groundwater",
                    level="ERROR",
                    message="date contains null/invalid entries",
                )
            )
        return issues
