"""
Fragmentation Undo Commands
=============================

Undoable commands for interactive point cloud editing:
- Delete selected points
- Crop to selection
- Merge fragments
- Split fragment
- Reclassify points
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .base_command import UndoableCommand

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DeletePointsCommand(UndoableCommand):
    """
    Delete selected points from the FragmentDataset.

    Stores the deleted indices and their data rows for undo.
    """

    def __init__(
        self,
        dataset,
        mask: np.ndarray,
        refresh_callback: Optional[Callable] = None,
        description: str = "Delete selected points",
    ):
        """
        Parameters
        ----------
        dataset : FragmentDataset (mutated in-place)
        mask : (N,) bool — True = points to DELETE
        refresh_callback : called after execute/undo to refresh renderer
        """
        super().__init__()
        self._dataset = dataset
        self._delete_mask = mask.copy()
        self._refresh = refresh_callback
        self._description = description

        # Store deleted data for undo
        self._deleted_indices = np.where(mask)[0]
        self._deleted_rows = dataset.fused_cloud[mask].copy()
        self._deleted_labels = None
        if dataset.fragment_labels is not None:
            self._deleted_labels = dataset.fragment_labels[mask].copy()

    @property
    def description(self) -> str:
        return f"{self._description} ({len(self._deleted_indices)} points)"

    def execute(self) -> None:
        keep = ~self._delete_mask
        self._dataset.fused_cloud = self._dataset.fused_cloud[keep]
        if self._dataset.fragment_labels is not None:
            self._dataset.fragment_labels = self._dataset.fragment_labels[keep]
        # Clear fragments list (invalidated by index change)
        self._dataset.fragments = []
        self._dataset.fsd_global = None

        logger.info("Deleted %d points", len(self._deleted_indices))
        if self._refresh:
            self._refresh()

    def undo(self) -> None:
        # Re-insert deleted rows at their original positions
        cloud = self._dataset.fused_cloud
        new_cloud = np.empty(
            (len(cloud) + len(self._deleted_rows), cloud.shape[1]),
            dtype=cloud.dtype,
        )
        # Build index map
        n_total = len(new_cloud)
        insert_mask = np.zeros(n_total, dtype=bool)
        insert_mask[self._deleted_indices] = True

        new_cloud[insert_mask] = self._deleted_rows
        new_cloud[~insert_mask] = cloud
        self._dataset.fused_cloud = new_cloud

        if self._deleted_labels is not None and self._dataset.fragment_labels is not None:
            labels = self._dataset.fragment_labels
            new_labels = np.full(n_total, -1, dtype=labels.dtype)
            new_labels[insert_mask] = self._deleted_labels
            new_labels[~insert_mask] = labels
            self._dataset.fragment_labels = new_labels

        logger.info("Undo: restored %d points", len(self._deleted_indices))
        if self._refresh:
            self._refresh()


class CropToSelectionCommand(UndoableCommand):
    """
    Crop the point cloud to keep only selected points.
    Equivalent to deleting the inverse of the selection.
    """

    def __init__(
        self,
        dataset,
        keep_mask: np.ndarray,
        refresh_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self._inner = DeletePointsCommand(
            dataset, ~keep_mask, refresh_callback,
            description="Crop to selection",
        )

    @property
    def description(self) -> str:
        return self._inner.description

    def execute(self) -> None:
        self._inner.execute()

    def undo(self) -> None:
        self._inner.undo()


class MergeFragmentsCommand(UndoableCommand):
    """Merge two or more fragments into one."""

    def __init__(
        self,
        dataset,
        fragment_ids: list,
        target_id: int,
        refresh_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self._dataset = dataset
        self._fragment_ids = fragment_ids
        self._target_id = target_id
        self._refresh = refresh_callback

        # Save old labels for undo
        labels = dataset.fragment_labels
        self._affected_mask = np.isin(labels, fragment_ids)
        self._old_labels = labels[self._affected_mask].copy()

    @property
    def description(self) -> str:
        return f"Merge fragments {self._fragment_ids} -> {self._target_id}"

    def execute(self) -> None:
        labels = self._dataset.fragment_labels
        for fid in self._fragment_ids:
            labels[labels == fid] = self._target_id

        self._dataset.fragments = []  # Invalidate
        self._dataset.fsd_global = None

        logger.info("Merged fragments %s -> %d", self._fragment_ids, self._target_id)
        if self._refresh:
            self._refresh()

    def undo(self) -> None:
        self._dataset.fragment_labels[self._affected_mask] = self._old_labels
        self._dataset.fragments = []
        self._dataset.fsd_global = None

        logger.info("Undo merge")
        if self._refresh:
            self._refresh()


class SplitFragmentCommand(UndoableCommand):
    """Split a fragment using a selection mask (selected points get new label)."""

    def __init__(
        self,
        dataset,
        fragment_id: int,
        split_mask: np.ndarray,
        refresh_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self._dataset = dataset
        self._fragment_id = fragment_id
        self._split_mask = split_mask.copy()
        self._refresh = refresh_callback
        self._new_id = int(dataset.fragment_labels.max() + 1) if dataset.fragment_labels is not None else 1

    @property
    def description(self) -> str:
        return f"Split fragment {self._fragment_id} -> {self._fragment_id}, {self._new_id}"

    def execute(self) -> None:
        labels = self._dataset.fragment_labels
        # Points that belong to this fragment AND are in the split mask get new ID
        affected = (labels == self._fragment_id) & self._split_mask
        labels[affected] = self._new_id

        self._dataset.fragments = []
        self._dataset.fsd_global = None

        logger.info("Split fragment %d: %d points -> new fragment %d",
                     self._fragment_id, int(affected.sum()), self._new_id)
        if self._refresh:
            self._refresh()

    def undo(self) -> None:
        labels = self._dataset.fragment_labels
        labels[labels == self._new_id] = self._fragment_id

        self._dataset.fragments = []
        self._dataset.fsd_global = None

        logger.info("Undo split fragment %d", self._fragment_id)
        if self._refresh:
            self._refresh()


class ReclassifyPointsCommand(UndoableCommand):
    """Reclassify selected points to a new fragment label (or -1 for noise)."""

    def __init__(
        self,
        dataset,
        mask: np.ndarray,
        new_label: int,
        refresh_callback: Optional[Callable] = None,
    ):
        super().__init__()
        self._dataset = dataset
        self._mask = mask.copy()
        self._new_label = new_label
        self._refresh = refresh_callback
        self._old_labels = dataset.fragment_labels[mask].copy()

    @property
    def description(self) -> str:
        label_name = "noise" if self._new_label == -1 else f"fragment {self._new_label}"
        return f"Reclassify {int(self._mask.sum())} points as {label_name}"

    def execute(self) -> None:
        self._dataset.fragment_labels[self._mask] = self._new_label
        self._dataset.fragments = []
        self._dataset.fsd_global = None
        if self._refresh:
            self._refresh()

    def undo(self) -> None:
        self._dataset.fragment_labels[self._mask] = self._old_labels
        self._dataset.fragments = []
        self._dataset.fsd_global = None
        if self._refresh:
            self._refresh()
