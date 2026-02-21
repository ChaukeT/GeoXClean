"""
Canonical Contact Extraction Algorithm

This module contains the single, canonical algorithm for extracting geological contacts
from drillhole composite data. All engines use this same algorithm to ensure
consistency and auditability.

Key Features:
- Deterministic: Same inputs always produce same outputs
- Comprehensive: Handles all contact types and edge cases
- Validated: Includes quality checks and warnings
- Documented: Full provenance tracking
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

from ..geology.data_contracts import Contact, ContactSet, CompositeCollection

logger = logging.getLogger(__name__)


def extract_contacts_from_composites(
    composites: CompositeCollection,
    min_contact_confidence: float = 0.8,
    max_contact_distance: float = 10.0,  # Max distance between adjacent intervals
) -> ContactSet:
    """
    Extract geological contacts from composite drillhole data.

    This is the canonical contact extraction algorithm used by all geological engines.
    It processes each hole independently, identifying where domain changes occur.

    Algorithm:
    1. Group intervals by hole and sort by depth
    2. For each consecutive pair of intervals in a hole:
       - Check if domains are different
       - Calculate contact position (midpoint)
       - Determine stratigraphic order based on depth
       - Create Contact object
    3. Validate contacts and filter low-quality ones
    4. Return immutable ContactSet

    Args:
        composites: Typed composite data
        min_contact_confidence: Minimum confidence threshold (0-1)
        max_contact_distance: Maximum distance between intervals to consider as contact

    Returns:
        Immutable ContactSet with all extracted contacts
    """
    logger.info("Starting canonical contact extraction")
    logger.info(f"Processing {len(composites.intervals)} composite intervals")

    contacts = []
    domain_pairs = set()

    # Process each hole separately
    for hole_id in composites.hole_ids:
        hole_contacts = _extract_contacts_from_hole(
            composites, hole_id, max_contact_distance
        )
        contacts.extend(hole_contacts)

        # Track domain pairs
        for contact in hole_contacts:
            pair = (contact.below_unit, contact.above_unit)
            domain_pairs.add(pair)

    logger.info(f"Extracted {len(contacts)} contacts from {len(composites.hole_ids)} holes")

    # Quality filtering
    high_quality_contacts = _filter_contacts_by_quality(
        contacts, min_contact_confidence
    )

    logger.info(f"After quality filtering: {len(high_quality_contacts)} contacts")

    # Create immutable ContactSet
    contact_set = ContactSet(
        contacts=frozenset(high_quality_contacts),
        domain_pairs=frozenset(domain_pairs)
    )

    # Log summary statistics
    _log_contact_statistics(contact_set)

    return contact_set


def _extract_contacts_from_hole(
    composites: CompositeCollection,
    hole_id: str,
    max_contact_distance: float
) -> List[Contact]:
    """
    Extract contacts from a single drillhole.

    Within each hole, contacts occur where consecutive intervals have different domains.
    The contact position is calculated as the midpoint between interval boundaries.
    """
    hole_intervals = composites.intervals_for_hole(hole_id)

    if len(hole_intervals) < 2:
        return []  # Need at least 2 intervals for a contact

    # Sort by depth (should already be sorted, but ensure)
    sorted_intervals = sorted(hole_intervals, key=lambda i: i.from_depth)

    contacts = []

    for i in range(len(sorted_intervals) - 1):
        current = sorted_intervals[i]
        next_interval = sorted_intervals[i + 1]

        # Check if domains are different
        if current.domain == next_interval.domain:
            continue  # No contact - same domain

        # Check distance between intervals
        gap = next_interval.from_depth - current.to_depth
        if gap > max_contact_distance:
            logger.warning(
                f"Hole {hole_id}: Large gap ({gap:.1f}m) between intervals "
                f"{current.domain}@{current.to_depth:.1f}m and "
                f"{next_interval.domain}@{next_interval.from_depth:.1f}m"
            )
            continue

        # Determine stratigraphic order
        # In drillholes, depth increases downward
        # Higher depth = older (assuming normal stratigraphy)
        if current.to_depth < next_interval.from_depth:
            # Normal case: current ends, next begins
            contact_depth = (current.to_depth + next_interval.from_depth) / 2

            # Stratigraphic order: shallower = younger, deeper = older
            # But we need to determine which domain is above vs below
            younger_domain = current.domain  # At shallower depth
            older_domain = next_interval.domain  # At deeper depth

        else:
            # Overlapping intervals - use midpoint
            contact_depth = (current.to_depth + next_interval.from_depth) / 2
            younger_domain = current.domain
            older_domain = next_interval.domain
            logger.debug(f"Hole {hole_id}: Overlapping intervals at depth {contact_depth:.1f}m")

        # Calculate contact position using hole trajectory
        # For now, use the position of the interval midpoint
        # TODO: Implement proper hole trajectory interpolation
        if current.z < next_interval.z:
            # Z increases upward
            contact_x = (current.x + next_interval.x) / 2
            contact_y = (current.y + next_interval.y) / 2
            contact_z = (current.z + next_interval.z) / 2
        else:
            # Use current interval position (simplified)
            contact_x = current.x
            contact_y = current.y
            contact_z = current.z

        # Determine contact type based on interval properties
        contact_type = _determine_contact_type(current, next_interval)

        # Calculate confidence based on data quality
        confidence = _calculate_contact_confidence(current, next_interval, gap)

        # Create contact
        contact = Contact(
            x=contact_x,
            y=contact_y,
            z=contact_z,
            above_unit=younger_domain,  # Above the contact = younger
            below_unit=older_domain,    # Below the contact = older
            hole_id=hole_id,
            depth=contact_depth,
            contact_type=contact_type,
            confidence=confidence,
            source="drillhole_composites"
        )

        contacts.append(contact)

    return contacts


def _determine_contact_type(interval1: "CompositeInterval", interval2: "CompositeInterval") -> str:
    """Determine the type of geological contact."""
    # Check for gradational contacts (based on attributes or metadata)
    # For now, assume all are sharp contacts
    # TODO: Implement logic based on assay gradients or logging data
    return "sharp"


def _calculate_contact_confidence(
    interval1: "CompositeInterval",
    interval2: "CompositeInterval",
    gap: float
) -> float:
    """Calculate confidence score for contact (0-1)."""
    confidence = 1.0

    # Penalize for gaps between intervals
    if gap > 0:
        gap_penalty = min(gap / 5.0, 0.5)  # Max 50% penalty for 5m gaps
        confidence -= gap_penalty

    # Penalize for very short intervals (possible compositing artifacts)
    min_length = min(interval1.length, interval2.length)
    if min_length < 0.5:  # Less than 0.5m
        length_penalty = (0.5 - min_length) / 0.5 * 0.2  # Max 20% penalty
        confidence -= length_penalty

    # Penalize if coordinates differ significantly (hole deviation)
    coord_diff = np.sqrt(
        (interval1.x - interval2.x)**2 +
        (interval1.y - interval2.y)**2 +
        (interval1.z - interval2.z)**2
    )
    if coord_diff > 10:  # More than 10m difference
        coord_penalty = min(coord_diff / 50.0, 0.3)  # Max 30% penalty
        confidence -= coord_penalty

    return max(0.0, confidence)


def _filter_contacts_by_quality(contacts: List[Contact], min_confidence: float) -> List[Contact]:
    """Filter contacts based on quality criteria."""
    filtered = []
    rejected = []

    for contact in contacts:
        if contact.confidence >= min_confidence:
            filtered.append(contact)
        else:
            rejected.append(contact)

    if rejected:
        logger.warning(
            f"Rejected {len(rejected)} low-quality contacts "
            f"(confidence < {min_confidence}). "
            f"Lowest confidence: {min(contact.confidence for contact in rejected):.2f}"
        )

    return filtered


def _log_contact_statistics(contact_set: ContactSet) -> None:
    """Log statistics about extracted contacts."""
    if not contact_set.contacts:
        logger.warning("No contacts extracted!")
        return

    # Count contacts per hole
    hole_counts = defaultdict(int)
    for contact in contact_set.contacts:
        hole_counts[contact.hole_id] += 1

    # Count contacts per domain pair
    pair_counts = defaultdict(int)
    for contact in contact_set.contacts:
        pair = (contact.below_unit, contact.above_unit)
        pair_counts[pair] += 1

    logger.info("Contact extraction statistics:")
    logger.info(f"  Total contacts: {len(contact_set.contacts)}")
    logger.info(f"  Holes with contacts: {len(hole_counts)}")
    logger.info(f"  Domain pairs: {len(contact_set.domain_pairs)}")

    # Log top holes and pairs
    if hole_counts:
        top_holes = sorted(hole_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top holes by contact count:")
        for hole, count in top_holes:
            logger.info(f"    {hole}: {count}")

    if pair_counts:
        top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top domain pairs by contact count:")
        for pair, count in top_pairs:
            logger.info(f"    {pair[0]} ↔ {pair[1]}: {count}")
