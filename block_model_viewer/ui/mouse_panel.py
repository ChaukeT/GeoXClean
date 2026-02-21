from __future__ import annotations



import logging

from typing import Optional, Any



from PyQt6.QtCore import Qt, QObject, QEvent, pyqtSignal, QTimer

from PyQt6.QtGui import QAction

from PyQt6.QtWidgets import (

    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QCheckBox,

    QRadioButton, QSlider, QPushButton, QFormLayout, QDoubleSpinBox,

    QButtonGroup, QFrame, QStyle

)



logger = logging.getLogger(__name__)



# Optional VTK import handling

try:

    import vtk

    VTK_AVAILABLE = True

except ImportError:

    VTK_AVAILABLE = False

from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

# =========================================================
# STYLING - Now using dynamic theme from modern_styles
# =========================================================
# Legacy DARK_STYLESHEET removed - use get_analysis_panel_stylesheet() instead

def get__placeholder_stylesheet() -> str:
    """Get  placeholder stylesheet stylesheet."""
    colors = get_theme_colors()
    return f"""

QWidget {{

    background-color: #2b2b2b;

    color: {colors.TEXT_PRIMARY};

    font-family: "Segoe UI", "Roboto", sans-serif;

    font-size: 10pt;

}}

QGroupBox {{

    border: 1px solid #3e3e3e;

    border-radius: 6px;

    margin-top: 22px;

    font-weight: bold;

}}

QGroupBox::title {{

    subcontrol-origin: margin;

    subcontrol-position: top left;

    padding: 0 5px;

    color: #007acc;

}}

QCheckBox, QRadioButton {{

    spacing: 8px;

}}

QCheckBox::indicator, QRadioButton::indicator {{

    width: 16px;

    height: 16px;

}}

QSlider::groove:horizontal {{

    border: 1px solid #3e3e3e;

    height: 6px;

    background: {colors.PANEL_BG};

    margin: 2px 0;

    border-radius: 3px;

}}

QSlider::handle:horizontal {{

    background: #555;

    border: 1px solid #555;

    width: 14px;

    height: 14px;

    margin: -5px 0;

    border-radius: 7px;

}}

QSlider::handle:horizontal:hover {{

    background: #007acc;

    border: 1px solid #007acc;

}}

QDoubleSpinBox {{

    background-color: {colors.PANEL_BG};

    border: 1px solid #3e3e3e;

    padding: 4px;

    border-radius: 3px;

}}

QPushButton {{

    background-color: #333333;

    border: 1px solid #555;

    border-radius: 4px;

    padding: 6px 12px;

    color: white;

}}

QPushButton:hover {{

    background-color: #444;

}}

QPushButton:pressed {{

    background-color: #222;

}}

"""



class _MouseEventFilter(QObject):

    """

    Intercept Qt wheel and double-click events to apply custom 

    navigation logic over the VTK window.

    """

    def __init__(self, owner: 'MousePanel'):

        super().__init__()

        self.owner = owner



    def eventFilter(self, watched: QObject, event: QEvent) -> bool:

        if not self.owner or not self.owner._settings_ready:

            return super().eventFilter(watched, event)



        try:

            # --- Custom Wheel Zoom ---

            if event.type() == QEvent.Type.Wheel:

                delta = event.angleDelta().y()

                if delta == 0: 

                    return False

                

                # Calculate Factor

                inv = self.owner.invert_wheel_cb.isChecked()

                base = max(1.01, self.owner.zoom_spin.value())

                

                direction = 1 if delta > 0 else -1

                if inv: direction *= -1

                

                factor = base if direction > 0 else (1.0 / base)

                

                # Execute Callback

                if hasattr(self.owner, '_zoom_callback'):

                    self.owner._zoom_callback(factor)

                    return True # Consume event



            # --- Double Click Fit ---

            if event.type() == QEvent.Type.MouseButtonDblClick:

                if self.owner.double_click_fit_cb.isChecked():

                    if hasattr(self.owner, '_fit_view_callback'):

                        self.owner._fit_view_callback()

                        return True # Consume event



        except Exception as e:

            logger.debug(f"MouseEventFilter error: {e}")

        

        return super().eventFilter(watched, event)





class MousePanel(QWidget):

    """

    Mouse & Interaction controls for the 3D view.

    Manages input modes (Trackball/Rubberband) and navigation sensitivity.

    """



    settings_changed = pyqtSignal(dict)



    def __init__(self, main_window, viewer_widget, config=None):

        super().__init__()

        self.main_window = main_window

        self.viewer_widget = viewer_widget

        self.config = config



        # State

        self._filter: Optional[_MouseEventFilter] = None

        self._attached_to = None

        self._settings_ready = False



        # Callbacks (Safety wrappers)

        self._zoom_callback = getattr(self.main_window, '_camera_zoom', lambda f: None)

        self._fit_view_callback = getattr(self.main_window, 'fit_to_view', lambda: None)

        self._reset_view_callback = getattr(self.main_window, 'reset_camera', lambda: None)



        self.setWindowTitle("Mouse Settings")

        self.setStyleSheet(get_analysis_panel_stylesheet())

        self._build_ui()

        self._load_defaults()



        # Defer attachment to allow VTK window to initialize

        QTimer.singleShot(600, self.attach_to_viewer)



    def _build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(10, 10, 10, 10)

        layout.setSpacing(15)



        # --- Group 1: Interaction Mode ---

        style_group = QGroupBox("Interaction Mode")

        style_layout = QVBoxLayout()

        

        self.btn_group = QButtonGroup(self)

        

        self.style_trackball = QRadioButton("Trackball (Orbit / Pan)")

        self.style_trackball.setToolTip("Standard 3D navigation using mouse buttons.")

        self.style_zoom = QRadioButton("Rubber-band Zoom")

        self.style_zoom.setToolTip("Click and drag to draw a box to zoom into.")

        self.style_pick = QRadioButton("Rubber-band Pick")

        self.style_pick.setToolTip("Click and drag to select items.")

        

        self.btn_group.addButton(self.style_trackball)

        self.btn_group.addButton(self.style_zoom)

        self.btn_group.addButton(self.style_pick)

        

        style_layout.addWidget(self.style_trackball)

        style_layout.addWidget(self.style_zoom)

        style_layout.addWidget(self.style_pick)

        style_group.setLayout(style_layout)

        layout.addWidget(style_group)



        # --- Group 2: Navigation Settings ---

        nav_group = QGroupBox("Navigation Behavior")

        nav_layout = QFormLayout()

        nav_layout.setSpacing(10)



        # Zoom Sensitivity (Slider + SpinBox)

        zoom_controls = QHBoxLayout()

        

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)

        self.zoom_slider.setRange(101, 150) # 1.01 to 1.50

        

        self.zoom_spin = QDoubleSpinBox()

        self.zoom_spin.setRange(1.01, 1.50)

        self.zoom_spin.setSingleStep(0.01)

        self.zoom_spin.setDecimals(2)

        self.zoom_spin.setFixedWidth(60)

        

        zoom_controls.addWidget(self.zoom_slider)

        zoom_controls.addWidget(self.zoom_spin)

        

        nav_layout.addRow("Zoom Speed:", zoom_controls)



        # Checkboxes

        self.invert_wheel_cb = QCheckBox("Invert Mouse Wheel")

        self.double_click_fit_cb = QCheckBox("Double-click to Fit View")

        

        nav_layout.addRow("", self.invert_wheel_cb)

        nav_layout.addRow("", self.double_click_fit_cb)

        

        nav_group.setLayout(nav_layout)

        layout.addWidget(nav_group)



        # --- Group 3: Camera Actions ---

        cam_group = QGroupBox("Camera Actions")

        cam_layout = QHBoxLayout()

        

        self.btn_reset = QPushButton("Reset Angle")

        self.btn_reset.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))

        

        self.btn_fit = QPushButton("Fit to Screen")

        self.btn_fit.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))

        

        cam_layout.addWidget(self.btn_reset)

        cam_layout.addWidget(self.btn_fit)

        cam_group.setLayout(cam_layout)

        layout.addWidget(cam_group)



        layout.addStretch()



        # --- Wiring ---

        # Sync Slider & SpinBox

        self.zoom_slider.valueChanged.connect(

            lambda v: self.zoom_spin.setValue(v / 100.0)

        )

        self.zoom_spin.valueChanged.connect(

            lambda v: self.zoom_slider.setValue(int(v * 100))

        )

        self.zoom_spin.valueChanged.connect(self._emit_settings)



        # Mode toggles

        self.style_trackball.toggled.connect(lambda v: v and self._apply_style('trackball'))

        self.style_zoom.toggled.connect(lambda v: v and self._apply_style('rubberband_zoom'))

        self.style_pick.toggled.connect(lambda v: v and self._apply_style('rubberband_pick'))

        

        # Prefs toggles

        self.invert_wheel_cb.toggled.connect(self._emit_settings)

        self.double_click_fit_cb.toggled.connect(self._emit_settings)

        

        # Buttons

        self.btn_reset.clicked.connect(self._reset_view)

        self.btn_fit.clicked.connect(self._fit_view)



        self._settings_ready = True



    # =========================================================

    # LOGIC

    # =========================================================



    def _apply_style(self, style_key: str):

        """Switches the interaction mode on the main window/interactor."""

        try:

            # 1. Notify Main Window (if methods exist)

            if style_key == 'trackball':

                if hasattr(self.main_window, 'set_mouse_mode_select'):

                    self.main_window.set_mouse_mode_select()

            

            elif style_key == 'rubberband_zoom':

                if hasattr(self.main_window, 'set_mouse_mode_zoom_box'):

                    self.main_window.set_mouse_mode_zoom_box()

            

            elif style_key == 'rubberband_pick':

                # 2. Direct VTK Interactor Manipulation (Fallback)

                interactor = self._get_interactor()

                if interactor and VTK_AVAILABLE:

                    interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())

        

        except Exception as e:

            logger.warning(f"Failed to apply interaction style '{style_key}': {e}")

            

        self._emit_settings()



    def _get_interactor(self) -> Optional[Any]:

        """Safely retrieves the VTK interactor from the viewer widget."""

        try:

            # Check standard PyVista/VTK structure

            if hasattr(self.viewer_widget, 'plotter'):

                return getattr(self.viewer_widget.plotter, 'interactor', None)

            # Fallback for raw VTK widgets

            if hasattr(self.viewer_widget, 'GetRenderWindow'):

                return self.viewer_widget.GetRenderWindow().GetInteractor()

        except Exception:

            pass

        return None



    def attach_to_viewer(self):

        """Installs the event filter onto the VTK Render Window Interactor."""

        interactor_widget = self._get_interactor()



        # If not ready, try again shortly

        if interactor_widget is None:

            QTimer.singleShot(1000, self.attach_to_viewer)

            return



        try:

            # Clean up old filter if re-attaching

            if self._attached_to and self._filter:

                try:

                    self._attached_to.removeEventFilter(self._filter)

                except RuntimeError:

                    pass # Object already deleted



            # Create and install new filter

            if self._filter is None:

                self._filter = _MouseEventFilter(self)

            

            interactor_widget.installEventFilter(self._filter)

            self._attached_to = interactor_widget

            logger.info("MousePanel: Successfully attached input filter.")

            

        except Exception as e:

            logger.error(f"MousePanel attach failed: {e}")



    def _emit_settings(self):

        """Collects current UI state and emits signal."""

        if not self._settings_ready: return

        

        mode = 'trackball'

        if self.style_zoom.isChecked(): mode = 'rubberband_zoom'

        elif self.style_pick.isChecked(): mode = 'rubberband_pick'

            

        settings = {

            'interaction_style': mode,

            'invert_wheel': self.invert_wheel_cb.isChecked(),

            'wheel_zoom_factor': self.zoom_spin.value(),

            'double_click_fit_view': self.double_click_fit_cb.isChecked()

        }

        self.settings_changed.emit(settings)

        

        # Save to config if present

        if self.config and hasattr(self.config, 'config'):

            self.config.config.setdefault('mouse', {}).update(settings)



    def _fit_view(self):

        self._fit_view_callback()



    def _reset_view(self):

        try:

            if hasattr(self.main_window, 'reset_view'):

                self.main_window.reset_view()

            else:

                self._reset_view_callback()

        except Exception:

            pass



    def _load_defaults(self):

        """Loads settings from config dictionary or sets defaults."""

        self._settings_ready = False # Block signals

        try:

            defaults = {}

            if self.config and hasattr(self.config, 'config'):

                defaults = self.config.config.get('mouse', {})



            # Style

            style = defaults.get('interaction_style', 'trackball')

            if style == 'rubberband_zoom': self.style_zoom.setChecked(True)

            elif style == 'rubberband_pick': self.style_pick.setChecked(True)

            else: self.style_trackball.setChecked(True)



            # Prefs

            self.invert_wheel_cb.setChecked(bool(defaults.get('invert_wheel', False)))

            self.double_click_fit_cb.setChecked(bool(defaults.get('double_click_fit_view', True)))

            

            # Zoom

            factor = float(defaults.get('wheel_zoom_factor', 1.2))

            self.zoom_spin.setValue(factor) # Slider updates automatically via signal connection



        except Exception as e:

            logger.warning(f"Error loading mouse defaults: {e}")

            self.style_trackball.setChecked(True)

            self.zoom_spin.setValue(1.20)

            self.double_click_fit_cb.setChecked(True)

            

        self._settings_ready = True





    def refresh_theme(self):
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())


# =========================================================

# TEST LAUNCHER

# =========================================================

if __name__ == "__main__":

    import sys

    from PyQt6.QtWidgets import QApplication, QMainWindow

    

    class MockWindow(QMainWindow):

        def fit_to_view(self): print("Mock: Fit View")

        def reset_camera(self): print("Mock: Reset Camera")

        def _camera_zoom(self, f): print(f"Mock: Zoom Factor {f}")

        def set_mouse_mode_select(self): print("Mock: Mode -> Trackball")

        def set_mouse_mode_zoom_box(self): print("Mock: Mode -> Box Zoom")



    class MockViewer:

        pass # No VTK installed for test



    app = QApplication(sys.argv)

    win = MockWindow()

    panel = MousePanel(win, MockViewer())

    panel.show()

    sys.exit(app.exec())
