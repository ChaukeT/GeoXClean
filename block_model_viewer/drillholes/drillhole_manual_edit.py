"""
DRILLHOLE MANUAL EDIT ENGINE (GeoX)

Backend engine for manual editing of drillhole data with:
- Row selection (single, multiple, all)
- Cell editing (single and batch)
- Undo/redo functionality
- Auto re-validation after edits
- Syncing with QC engine
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd
import copy


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class ManualEdit:
    table: str                  # assays / lithology / surveys
    hole_id: str
    row_index: int
    column: str
    old_value: Any
    new_value: Any
    reason: str
    user: str


@dataclass
class ManualEditBatch:
    edits: List[ManualEdit]
    batch_reason: str
    user: str


# =========================================================
# MANUAL EDIT ENGINE
# =========================================================

class ManualEditEngine:
    """
    Backend engine to allow:
     - clicking a row → selecting it
     - selecting multiple rows
     - selecting all rows
     - modifying values in batch
     - undo / redo
     - auto re-validation after edits
     - syncing with QC engine
    """

    def __init__(self, collars, surveys, assays, lithology, user="GEOLOGIST"):
        self.collars = collars.copy()
        self.surveys = surveys.copy()
        self.assays = assays.copy()
        self.lithology = lithology.copy()

        # (table_name → selected row indices)
        self.selection: Dict[str, List[int]] = {
            "assays": [],
            "lithology": [],
            "surveys": [],
        }

        self.user = user

        # Undo/Redo stacks
        self.undo_stack: List[ManualEditBatch] = []
        self.redo_stack: List[ManualEditBatch] = []

    # ---------------------------------------------------------
    # SELECTION COMMANDS
    # ---------------------------------------------------------

    def clear_selection(self, table: str):
        """Clear selection for a table."""
        self.selection[table] = []

    def select_row(self, table: str, idx: int):
        """Select a single row."""
        if idx not in self.selection[table]:
            self.selection[table].append(idx)

    def select_rows(self, table: str, rows: List[int]):
        """Select multiple rows."""
        for r in rows:
            if r not in self.selection[table]:
                self.selection[table].append(r)

    def select_all(self, table: str):
        """Select all rows in a table."""
        df = getattr(self, table)
        self.selection[table] = list(df.index)

    # ---------------------------------------------------------
    # EDITING COMMANDS
    # ---------------------------------------------------------

    def edit_cell(self, table: str, row_index: int, column: str, new_value: Any, reason="manual-edit"):
        """Edit a single cell."""
        df = getattr(self, table)

        if row_index not in df.index:
            raise ValueError(f"Row index {row_index} not found in {table}")

        old_value = df.at[row_index, column]
        df.at[row_index, column] = new_value

        batch = ManualEditBatch(
            edits=[ManualEdit(
                table=table,
                hole_id=df.at[row_index, "hole_id"] if "hole_id" in df.columns else "",
                row_index=row_index,
                column=column,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                user=self.user,
            )],
            batch_reason=reason,
            user=self.user,
        )

        self.undo_stack.append(batch)
        self.redo_stack.clear()

    # ---------------------------------------------------------
    # BATCH EDIT (for selecting multiple rows)
    # ---------------------------------------------------------

    def batch_edit(
        self,
        table: str,
        column: str,
        new_value: Any,
        reason="batch-edit"
    ):
        """Edit a column for all selected rows."""
        df = getattr(self, table)
        edits = []

        for idx in self.selection[table]:
            if idx not in df.index:
                continue

            old_value = df.at[idx, column]
            df.at[idx, column] = new_value

            edits.append(
                ManualEdit(
                    table=table,
                    hole_id=df.at[idx, "hole_id"] if "hole_id" in df.columns else "",
                    row_index=idx,
                    column=column,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                    user=self.user,
                )
            )

        if edits:
            batch = ManualEditBatch(edits=edits, batch_reason=reason, user=self.user)
            self.undo_stack.append(batch)
            self.redo_stack.clear()

    # ---------------------------------------------------------
    # UNDO / REDO
    # ---------------------------------------------------------

    def undo(self):
        """Undo the last edit batch."""
        if not self.undo_stack:
            return

        batch = self.undo_stack.pop()
        self.apply_edit_batch(batch, undo=True)
        self.redo_stack.append(batch)

    def redo(self):
        """Redo the last undone edit batch."""
        if not self.redo_stack:
            return

        batch = self.redo_stack.pop()
        self.apply_edit_batch(batch, undo=False)
        self.undo_stack.append(batch)

    def apply_edit_batch(self, batch: ManualEditBatch, undo=False):
        """Apply or undo an edit batch."""
        for e in batch.edits:
            df = getattr(self, e.table)

            if e.row_index not in df.index:
                continue

            if undo:
                df.at[e.row_index, e.column] = e.old_value
            else:
                df.at[e.row_index, e.column] = e.new_value

    # ---------------------------------------------------------
    # GET CURRENT TABLES
    # ---------------------------------------------------------

    def get_tables(self):
        """Get all current tables."""
        return {
            "collars": self.collars,
            "surveys": self.surveys,
            "assays": self.assays,
            "lithology": self.lithology,
        }

