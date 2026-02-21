"""
Layout document model for GeoX Layout Composer.

Dataclasses defining the structure of layout documents including pages,
items (viewport, legend, scale bar, north arrow, text, image, metadata),
and serialization support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


# ============================================================================
# Enumerations
# ============================================================================

class PageSize(Enum):
    """Standard page size presets."""
    A4 = "A4"           # 210 x 297 mm
    A3 = "A3"           # 297 x 420 mm
    A2 = "A2"           # 420 x 594 mm
    LETTER = "Letter"   # 215.9 x 279.4 mm
    CUSTOM = "Custom"


class PageOrientation(Enum):
    """Page orientation."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


# Standard page dimensions in mm (width x height for portrait)
PAGE_DIMENSIONS = {
    PageSize.A4: (210.0, 297.0),
    PageSize.A3: (297.0, 420.0),
    PageSize.A2: (420.0, 594.0),
    PageSize.LETTER: (215.9, 279.4),
}


# ============================================================================
# Page Specification
# ============================================================================

@dataclass
class PageSpec:
    """Page configuration in millimeters."""
    size: PageSize = PageSize.A4
    orientation: PageOrientation = PageOrientation.LANDSCAPE
    width_mm: float = 297.0   # Default: A4 landscape
    height_mm: float = 210.0
    margin_left_mm: float = 10.0
    margin_right_mm: float = 10.0
    margin_top_mm: float = 10.0
    margin_bottom_mm: float = 10.0

    def __post_init__(self):
        """Update dimensions based on size and orientation if using preset."""
        if self.size != PageSize.CUSTOM and self.size in PAGE_DIMENSIONS:
            base_w, base_h = PAGE_DIMENSIONS[self.size]
            if self.orientation == PageOrientation.LANDSCAPE:
                self.width_mm = base_h
                self.height_mm = base_w
            else:
                self.width_mm = base_w
                self.height_mm = base_h

    @property
    def content_width_mm(self) -> float:
        """Width available for content (excluding margins)."""
        return self.width_mm - self.margin_left_mm - self.margin_right_mm

    @property
    def content_height_mm(self) -> float:
        """Height available for content (excluding margins)."""
        return self.height_mm - self.margin_top_mm - self.margin_bottom_mm

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "size": self.size.value,
            "orientation": self.orientation.value,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "margin_left_mm": self.margin_left_mm,
            "margin_right_mm": self.margin_right_mm,
            "margin_top_mm": self.margin_top_mm,
            "margin_bottom_mm": self.margin_bottom_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageSpec":
        """Deserialize from dictionary."""
        spec = cls.__new__(cls)
        spec.size = PageSize(data.get("size", "A4"))
        spec.orientation = PageOrientation(data.get("orientation", "landscape"))
        spec.width_mm = data.get("width_mm", 297.0)
        spec.height_mm = data.get("height_mm", 210.0)
        spec.margin_left_mm = data.get("margin_left_mm", 10.0)
        spec.margin_right_mm = data.get("margin_right_mm", 10.0)
        spec.margin_top_mm = data.get("margin_top_mm", 10.0)
        spec.margin_bottom_mm = data.get("margin_bottom_mm", 10.0)
        return spec


# ============================================================================
# Layout Items
# ============================================================================

def _generate_id() -> str:
    """Generate a short unique identifier."""
    return str(uuid.uuid4())[:8]


@dataclass
class LayoutItem:
    """Base class for all layout items."""
    id: str = field(default_factory=_generate_id)
    name: str = ""
    item_type: str = "base"
    x_mm: float = 0.0
    y_mm: float = 0.0
    width_mm: float = 50.0
    height_mm: float = 50.0
    rotation_deg: float = 0.0
    visible: bool = True
    locked: bool = False
    z_order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "item_type": self.item_type,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "rotation_deg": self.rotation_deg,
            "visible": self.visible,
            "locked": self.locked,
            "z_order": self.z_order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutItem":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            item_type=data.get("item_type", "base"),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 50.0),
            height_mm=data.get("height_mm", 50.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
        )


@dataclass
class ViewportItem(LayoutItem):
    """3D viewport captured from the main viewer."""
    item_type: str = field(default="viewport", init=False)
    camera_state: Optional[Dict[str, Any]] = None
    legend_state: Optional[Dict[str, Any]] = None
    layers_snapshot: Optional[List[str]] = None
    background_color: str = "#1a1a1a"
    show_axes: bool = False

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "camera_state": self.camera_state,
            "legend_state": self.legend_state,
            "layers_snapshot": self.layers_snapshot,
            "background_color": self.background_color,
            "show_axes": self.show_axes,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViewportItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 150.0),
            height_mm=data.get("height_mm", 100.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            camera_state=data.get("camera_state"),
            legend_state=data.get("legend_state"),
            layers_snapshot=data.get("layers_snapshot"),
            background_color=data.get("background_color", "#1a1a1a"),
            show_axes=data.get("show_axes", False),
        )


@dataclass
class LegendItem(LayoutItem):
    """Legend from current visualization state."""
    item_type: str = field(default="legend", init=False)
    legend_state: Optional[Dict[str, Any]] = None
    orientation: str = "vertical"  # vertical or horizontal
    font_size: int = 10
    font_family: str = "Segoe UI"
    show_title: bool = True
    background_color: str = "#1e1e1e"
    text_color: str = "#f0f0f0"

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "legend_state": self.legend_state,
            "orientation": self.orientation,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "show_title": self.show_title,
            "background_color": self.background_color,
            "text_color": self.text_color,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegendItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 40.0),
            height_mm=data.get("height_mm", 80.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            legend_state=data.get("legend_state"),
            orientation=data.get("orientation", "vertical"),
            font_size=data.get("font_size", 10),
            font_family=data.get("font_family", "Segoe UI"),
            show_title=data.get("show_title", True),
            background_color=data.get("background_color", "#1e1e1e"),
            text_color=data.get("text_color", "#f0f0f0"),
        )


@dataclass
class ScaleBarItem(LayoutItem):
    """Scale bar showing map units."""
    item_type: str = field(default="scale_bar", init=False)
    units: str = "m"
    style: str = "alternating"  # alternating, single, stepped
    num_segments: int = 4
    font_size: int = 10
    font_family: str = "Segoe UI"
    bar_height_mm: float = 3.0
    background_color: Optional[str] = None
    text_color: str = "#000000"
    bar_color: str = "#000000"
    alt_bar_color: str = "#ffffff"
    # Computed at render time based on viewport scale
    scale_value: Optional[float] = None  # meters per mm at viewport center
    is_approximate: bool = False  # True if perspective projection

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "units": self.units,
            "style": self.style,
            "num_segments": self.num_segments,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "bar_height_mm": self.bar_height_mm,
            "background_color": self.background_color,
            "text_color": self.text_color,
            "bar_color": self.bar_color,
            "alt_bar_color": self.alt_bar_color,
            "scale_value": self.scale_value,
            "is_approximate": self.is_approximate,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleBarItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 50.0),
            height_mm=data.get("height_mm", 10.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            units=data.get("units", "m"),
            style=data.get("style", "alternating"),
            num_segments=data.get("num_segments", 4),
            font_size=data.get("font_size", 10),
            font_family=data.get("font_family", "Segoe UI"),
            bar_height_mm=data.get("bar_height_mm", 3.0),
            background_color=data.get("background_color"),
            text_color=data.get("text_color", "#000000"),
            bar_color=data.get("bar_color", "#000000"),
            alt_bar_color=data.get("alt_bar_color", "#ffffff"),
            scale_value=data.get("scale_value"),
            is_approximate=data.get("is_approximate", False),
        )


@dataclass
class NorthArrowItem(LayoutItem):
    """North arrow indicator."""
    item_type: str = field(default="north_arrow", init=False)
    style: str = "simple"  # simple, compass, fancy
    rotation_override: Optional[float] = None  # None = auto from camera azimuth
    fill_color: str = "#000000"
    outline_color: str = "#ffffff"
    show_label: bool = True
    label_text: str = "N"
    font_size: int = 12
    font_family: str = "Segoe UI"
    # Status flags
    has_crs: bool = True  # If False, show warning
    warning_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "style": self.style,
            "rotation_override": self.rotation_override,
            "fill_color": self.fill_color,
            "outline_color": self.outline_color,
            "show_label": self.show_label,
            "label_text": self.label_text,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "has_crs": self.has_crs,
            "warning_message": self.warning_message,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NorthArrowItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 15.0),
            height_mm=data.get("height_mm", 20.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            style=data.get("style", "simple"),
            rotation_override=data.get("rotation_override"),
            fill_color=data.get("fill_color", "#000000"),
            outline_color=data.get("outline_color", "#ffffff"),
            show_label=data.get("show_label", True),
            label_text=data.get("label_text", "N"),
            font_size=data.get("font_size", 12),
            font_family=data.get("font_family", "Segoe UI"),
            has_crs=data.get("has_crs", True),
            warning_message=data.get("warning_message"),
        )


@dataclass
class TextItem(LayoutItem):
    """Free text annotation."""
    item_type: str = field(default="text", init=False)
    text: str = ""
    font_family: str = "Segoe UI"
    font_size: int = 12
    font_bold: bool = False
    font_italic: bool = False
    text_color: str = "#000000"
    background_color: Optional[str] = None
    alignment: str = "left"  # left, center, right
    vertical_alignment: str = "top"  # top, middle, bottom
    word_wrap: bool = True
    padding_mm: float = 2.0
    border_color: Optional[str] = None
    border_width_mm: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "text": self.text,
            "font_family": self.font_family,
            "font_size": self.font_size,
            "font_bold": self.font_bold,
            "font_italic": self.font_italic,
            "text_color": self.text_color,
            "background_color": self.background_color,
            "alignment": self.alignment,
            "vertical_alignment": self.vertical_alignment,
            "word_wrap": self.word_wrap,
            "padding_mm": self.padding_mm,
            "border_color": self.border_color,
            "border_width_mm": self.border_width_mm,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 60.0),
            height_mm=data.get("height_mm", 15.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            text=data.get("text", ""),
            font_family=data.get("font_family", "Segoe UI"),
            font_size=data.get("font_size", 12),
            font_bold=data.get("font_bold", False),
            font_italic=data.get("font_italic", False),
            text_color=data.get("text_color", "#000000"),
            background_color=data.get("background_color"),
            alignment=data.get("alignment", "left"),
            vertical_alignment=data.get("vertical_alignment", "top"),
            word_wrap=data.get("word_wrap", True),
            padding_mm=data.get("padding_mm", 2.0),
            border_color=data.get("border_color"),
            border_width_mm=data.get("border_width_mm", 0.5),
        )


@dataclass
class ImageItem(LayoutItem):
    """Image/logo placement."""
    item_type: str = field(default="image", init=False)
    image_path: Optional[str] = None
    image_data_base64: Optional[str] = None  # For embedded images
    maintain_aspect: bool = True
    opacity: float = 1.0
    border_color: Optional[str] = None
    border_width_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "image_path": self.image_path,
            "image_data_base64": self.image_data_base64,
            "maintain_aspect": self.maintain_aspect,
            "opacity": self.opacity,
            "border_color": self.border_color,
            "border_width_mm": self.border_width_mm,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 30.0),
            height_mm=data.get("height_mm", 30.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            image_path=data.get("image_path"),
            image_data_base64=data.get("image_data_base64"),
            maintain_aspect=data.get("maintain_aspect", True),
            opacity=data.get("opacity", 1.0),
            border_color=data.get("border_color"),
            border_width_mm=data.get("border_width_mm", 0.0),
        )


@dataclass
class MetadataItem(LayoutItem):
    """Dynamic metadata block (project info, date, etc.)."""
    item_type: str = field(default="metadata", init=False)
    fields: List[str] = field(default_factory=lambda: [
        "project_name", "date", "author", "crs", "software_version"
    ])
    font_size: int = 9
    font_family: str = "Segoe UI"
    show_labels: bool = True
    label_width_mm: float = 25.0
    text_color: str = "#000000"
    background_color: Optional[str] = None
    line_spacing: float = 1.2
    # Custom field values (override dynamic values)
    custom_values: Dict[str, str] = field(default_factory=dict)

    # Available metadata fields
    AVAILABLE_FIELDS = [
        "project_name",
        "date",
        "author",
        "crs",
        "software_version",
        "dataset_name",
        "domain",
        "run_id",
        "variogram_signature",
        "export_dpi",
        "page_size",
    ]

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "fields": self.fields,
            "font_size": self.font_size,
            "font_family": self.font_family,
            "show_labels": self.show_labels,
            "label_width_mm": self.label_width_mm,
            "text_color": self.text_color,
            "background_color": self.background_color,
            "line_spacing": self.line_spacing,
            "custom_values": self.custom_values,
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataItem":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", ""),
            x_mm=data.get("x_mm", 0.0),
            y_mm=data.get("y_mm", 0.0),
            width_mm=data.get("width_mm", 70.0),
            height_mm=data.get("height_mm", 40.0),
            rotation_deg=data.get("rotation_deg", 0.0),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
            z_order=data.get("z_order", 0),
            fields=data.get("fields", ["project_name", "date", "author"]),
            font_size=data.get("font_size", 9),
            font_family=data.get("font_family", "Segoe UI"),
            show_labels=data.get("show_labels", True),
            label_width_mm=data.get("label_width_mm", 25.0),
            text_color=data.get("text_color", "#000000"),
            background_color=data.get("background_color"),
            line_spacing=data.get("line_spacing", 1.2),
            custom_values=data.get("custom_values", {}),
        )


# ============================================================================
# Layout Document
# ============================================================================

# Item type registry for polymorphic deserialization
ITEM_TYPE_REGISTRY: Dict[str, type] = {
    "base": LayoutItem,
    "viewport": ViewportItem,
    "legend": LegendItem,
    "scale_bar": ScaleBarItem,
    "north_arrow": NorthArrowItem,
    "text": TextItem,
    "image": ImageItem,
    "metadata": MetadataItem,
}


@dataclass
class LayoutDocument:
    """Complete layout document."""
    version: str = "1.0.0"
    name: str = "Untitled Layout"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    page: PageSpec = field(default_factory=PageSpec)
    items: List[LayoutItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Footer settings
    show_footer: bool = True
    footer_text: str = "Generated by GeoX"

    def add_item(self, item: LayoutItem) -> None:
        """Add an item to the layout."""
        if self.items:
            item.z_order = max(i.z_order for i in self.items) + 1
        self.items.append(item)
        self.modified = datetime.now().isoformat()

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by ID."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                self.modified = datetime.now().isoformat()
                return True
        return False

    def get_item(self, item_id: str) -> Optional[LayoutItem]:
        """Get an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_items_sorted(self) -> List[LayoutItem]:
        """Get items sorted by z_order (bottom to top)."""
        return sorted(self.items, key=lambda x: x.z_order)

    def move_item_to_front(self, item_id: str) -> None:
        """Move an item to the front (highest z_order)."""
        item = self.get_item(item_id)
        if item and self.items:
            item.z_order = max(i.z_order for i in self.items) + 1
            self.modified = datetime.now().isoformat()

    def move_item_to_back(self, item_id: str) -> None:
        """Move an item to the back (lowest z_order)."""
        item = self.get_item(item_id)
        if item and self.items:
            min_z = min(i.z_order for i in self.items)
            item.z_order = min_z - 1
            self.modified = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "page": self.page.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
            "show_footer": self.show_footer,
            "footer_text": self.footer_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutDocument":
        """Deserialize from dictionary."""
        items = []
        for item_data in data.get("items", []):
            item_type = item_data.get("item_type", "base")
            item_cls = ITEM_TYPE_REGISTRY.get(item_type, LayoutItem)
            items.append(item_cls.from_dict(item_data))

        return cls(
            version=data.get("version", "1.0.0"),
            name=data.get("name", "Untitled Layout"),
            created=data.get("created", datetime.now().isoformat()),
            modified=data.get("modified", datetime.now().isoformat()),
            page=PageSpec.from_dict(data.get("page", {})),
            items=items,
            metadata=data.get("metadata", {}),
            show_footer=data.get("show_footer", True),
            footer_text=data.get("footer_text", "Generated by GeoX"),
        )

    def create_default_layout(self) -> None:
        """Populate with a default layout template."""
        self.items.clear()

        page = self.page
        margin = page.margin_left_mm

        # Title at top center
        title = TextItem(
            name="Title",
            text="Layout Title",
            x_mm=margin,
            y_mm=margin,
            width_mm=page.content_width_mm,
            height_mm=15.0,
            font_size=18,
            font_bold=True,
            alignment="center",
            z_order=100,
        )
        self.items.append(title)

        # Viewport in center-left area
        viewport = ViewportItem(
            name="Main Viewport",
            x_mm=margin,
            y_mm=margin + 20,
            width_mm=page.content_width_mm * 0.7,
            height_mm=page.content_height_mm - 50,
            z_order=0,
        )
        self.items.append(viewport)

        # Legend on right side
        legend = LegendItem(
            name="Legend",
            x_mm=margin + page.content_width_mm * 0.72,
            y_mm=margin + 20,
            width_mm=page.content_width_mm * 0.26,
            height_mm=80.0,
            z_order=10,
        )
        self.items.append(legend)

        # Scale bar at bottom left
        scale_bar = ScaleBarItem(
            name="Scale Bar",
            x_mm=margin,
            y_mm=page.height_mm - margin - 15,
            width_mm=60.0,
            height_mm=10.0,
            z_order=20,
        )
        self.items.append(scale_bar)

        # North arrow below legend
        north = NorthArrowItem(
            name="North Arrow",
            x_mm=margin + page.content_width_mm * 0.72 + 10,
            y_mm=margin + 110,
            width_mm=20.0,
            height_mm=25.0,
            z_order=30,
        )
        self.items.append(north)

        # Metadata at bottom right (sized for 5 fields)
        metadata = MetadataItem(
            name="Metadata",
            x_mm=page.width_mm - margin - 90,
            y_mm=page.height_mm - margin - 55,
            width_mm=90.0,
            height_mm=50.0,
            z_order=40,
        )
        self.items.append(metadata)

        self.modified = datetime.now().isoformat()
