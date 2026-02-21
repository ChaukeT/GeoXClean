"""
Data models for the multi-element legend system.

Provides dataclasses for representing individual legend elements (continuous or discrete)
and a container for managing multiple legend elements simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Tuple, Any
from enum import Enum
import uuid


class LegendElementType(Enum):
    """Type of legend element."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class LegendElement:
    """
    Individual legend entry that can be continuous or discrete.

    Each element represents a single layer/property combination in the legend.
    """
    id: str                                    # Unique identifier (e.g., "block_model.Au_ppm")
    element_type: LegendElementType            # CONTINUOUS or DISCRETE
    title: str                                 # Display title
    source_layer: str                          # Layer this element represents
    source_property: Optional[str] = None      # Property name (for block model props)
    visible: bool = True                       # Whether element is shown in legend

    # Continuous-specific fields
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cmap_name: str = "viridis"
    log_scale: bool = False
    reverse: bool = False
    tick_count: int = 5

    # Discrete-specific fields
    categories: List[Union[str, int, float]] = field(default_factory=list)
    category_colors: Dict[Union[str, int, float], Tuple[float, float, float, float]] = field(default_factory=dict)
    category_labels: Dict[Union[str, int, float], str] = field(default_factory=dict)
    category_visible: Dict[Union[str, int, float], bool] = field(default_factory=dict)

    @classmethod
    def create_id(cls, layer: str, property_name: Optional[str] = None) -> str:
        """Generate a unique element ID from layer and property."""
        if property_name:
            return f"{layer}.{property_name}"
        return layer

    @classmethod
    def create_continuous(
        cls,
        layer: str,
        property_name: str,
        title: str,
        vmin: float,
        vmax: float,
        cmap_name: str = "viridis",
        log_scale: bool = False,
        reverse: bool = False,
    ) -> "LegendElement":
        """Factory method to create a continuous legend element."""
        return cls(
            id=cls.create_id(layer, property_name),
            element_type=LegendElementType.CONTINUOUS,
            title=title,
            source_layer=layer,
            source_property=property_name,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            log_scale=log_scale,
            reverse=reverse,
        )

    @classmethod
    def create_discrete(
        cls,
        layer: str,
        property_name: Optional[str],
        title: str,
        categories: List[Union[str, int, float]],
        category_colors: Optional[Dict[Union[str, int, float], Tuple[float, float, float, float]]] = None,
        category_labels: Optional[Dict[Union[str, int, float], str]] = None,
    ) -> "LegendElement":
        """Factory method to create a discrete legend element."""
        elem = cls(
            id=cls.create_id(layer, property_name),
            element_type=LegendElementType.DISCRETE,
            title=title,
            source_layer=layer,
            source_property=property_name,
            categories=list(categories),
            category_colors=category_colors or {},
            category_labels=category_labels or {},
        )
        # Initialize all categories as visible
        elem.category_visible = {cat: True for cat in categories}
        return elem

    def to_dict(self) -> Dict[str, Any]:
        """Serialize element to dictionary for persistence."""
        data = {
            "id": self.id,
            "element_type": self.element_type.value,
            "title": self.title,
            "source_layer": self.source_layer,
            "source_property": self.source_property,
            "visible": self.visible,
        }

        if self.element_type == LegendElementType.CONTINUOUS:
            data.update({
                "vmin": self.vmin,
                "vmax": self.vmax,
                "cmap_name": self.cmap_name,
                "log_scale": self.log_scale,
                "reverse": self.reverse,
                "tick_count": self.tick_count,
            })
        else:
            # Convert category keys to strings for JSON serialization
            data.update({
                "categories": [str(c) for c in self.categories],
                "category_colors": {str(k): list(v) for k, v in self.category_colors.items()},
                "category_labels": {str(k): v for k, v in self.category_labels.items()},
                "category_visible": {str(k): v for k, v in self.category_visible.items()},
            })

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegendElement":
        """Deserialize element from dictionary."""
        element_type = LegendElementType(data["element_type"])

        elem = cls(
            id=data["id"],
            element_type=element_type,
            title=data["title"],
            source_layer=data["source_layer"],
            source_property=data.get("source_property"),
            visible=data.get("visible", True),
        )

        if element_type == LegendElementType.CONTINUOUS:
            elem.vmin = data.get("vmin")
            elem.vmax = data.get("vmax")
            elem.cmap_name = data.get("cmap_name", "viridis")
            elem.log_scale = data.get("log_scale", False)
            elem.reverse = data.get("reverse", False)
            elem.tick_count = data.get("tick_count", 5)
        else:
            elem.categories = data.get("categories", [])
            # Restore category colors as tuples
            elem.category_colors = {
                k: tuple(v) for k, v in data.get("category_colors", {}).items()
            }
            elem.category_labels = data.get("category_labels", {})
            elem.category_visible = data.get("category_visible", {})

        return elem


@dataclass
class MultiLegendConfig:
    """
    Container for multiple legend elements.

    Manages a collection of legend elements that can be displayed simultaneously,
    with support for add/remove operations and persistence.
    """
    elements: List[LegendElement] = field(default_factory=list)
    layout: str = "vertical"                   # "vertical" or "horizontal"
    max_visible: int = 10                      # Max elements shown before scrolling

    def add_element(self, element: LegendElement) -> None:
        """
        Add or update a legend element.

        If an element with the same ID exists, it will be replaced.
        """
        # Remove existing element with same id
        self.elements = [e for e in self.elements if e.id != element.id]
        self.elements.append(element)

    def remove_element(self, element_id: str) -> bool:
        """
        Remove element by ID.

        Returns True if element was found and removed, False otherwise.
        """
        original_len = len(self.elements)
        self.elements = [e for e in self.elements if e.id != element_id]
        return len(self.elements) < original_len

    def get_element(self, element_id: str) -> Optional[LegendElement]:
        """Get element by ID, or None if not found."""
        return next((e for e in self.elements if e.id == element_id), None)

    def set_visibility(self, element_id: str, visible: bool) -> bool:
        """
        Set visibility for an element.

        Returns True if element was found and updated, False otherwise.
        """
        elem = self.get_element(element_id)
        if elem:
            elem.visible = visible
            return True
        return False

    def get_visible_elements(self) -> List[LegendElement]:
        """Get all visible elements."""
        return [e for e in self.elements if e.visible]

    def clear(self) -> None:
        """Remove all elements."""
        self.elements.clear()

    def has_element(self, element_id: str) -> bool:
        """Check if an element with the given ID exists."""
        return any(e.id == element_id for e in self.elements)

    def move_element(self, element_id: str, new_index: int) -> bool:
        """
        Move element to a new position in the list.

        Returns True if element was found and moved, False otherwise.
        """
        elem = self.get_element(element_id)
        if elem is None:
            return False

        self.elements.remove(elem)
        new_index = max(0, min(new_index, len(self.elements)))
        self.elements.insert(new_index, elem)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary for persistence."""
        return {
            "elements": [e.to_dict() for e in self.elements],
            "layout": self.layout,
            "max_visible": self.max_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiLegendConfig":
        """Deserialize config from dictionary."""
        config = cls(
            layout=data.get("layout", "vertical"),
            max_visible=data.get("max_visible", 10),
        )

        for elem_data in data.get("elements", []):
            try:
                elem = LegendElement.from_dict(elem_data)
                config.elements.append(elem)
            except (KeyError, ValueError) as e:
                # Skip invalid elements
                pass

        return config
