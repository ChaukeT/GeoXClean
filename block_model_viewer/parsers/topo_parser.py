"""
Topographic surface parser for GeoTIFF, ASCII Grid, and Surfer Grid formats.

Converts raster elevation data into PyVista StructuredGrid surface meshes
suitable for 3D visualization in GeoX.
"""

import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.asc', '.grd']


class TopoParser:
    """Parser for topographic surface formats -> PyVista StructuredGrid."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path, stride: int = 1) -> pv.StructuredGrid:
        """
        Parse a topographic file into a PyVista StructuredGrid.

        Args:
            file_path: Path to the topo file.
            stride: Downsample factor (1 = full resolution, 2 = every 2nd pixel, etc.).

        Returns:
            pv.StructuredGrid with elevation as point data 'Elevation'.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext in ('.tif', '.tiff'):
            return self._parse_geotiff(file_path, stride)
        elif ext == '.asc':
            return self._parse_asc(file_path, stride)
        elif ext == '.grd':
            return self._parse_grd(file_path, stride)
        else:
            raise ValueError(f"Unsupported topographic format: {ext}")

    # ── GeoTIFF ──────────────────────────────────────────────────────────

    def _parse_geotiff(self, path: Path, stride: int) -> pv.StructuredGrid:
        """Parse GeoTIFF raster using tifffile."""
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required for GeoTIFF import. Install with: pip install tifffile")

        with tifffile.TiffFile(str(path)) as tif:
            image = tif.pages[0].asarray()
            tags = tif.pages[0].tags

        # Handle multi-band: take first band
        if image.ndim == 3:
            image = image[0]

        nrows, ncols = image.shape
        logger.info(f"GeoTIFF: {ncols}x{nrows} pixels")

        # Extract geotransform from TIFF tags
        origin_x, origin_y, cell_w, cell_h = self._extract_geotransform(tags, ncols, nrows)

        # Handle nodata
        nodata = self._get_nodata(tags)
        if nodata is not None:
            mask = np.isclose(image, nodata, rtol=1e-5)
            image = image.astype(np.float64)
            image[mask] = np.nan

        return self._build_structured_grid(image, origin_x, origin_y, cell_w, cell_h, stride)

    def _extract_geotransform(self, tags, ncols: int, nrows: int) -> Tuple[float, float, float, float]:
        """Extract origin and pixel size from TIFF tags."""
        cell_w = cell_h = 1.0
        origin_x = origin_y = 0.0

        # ModelPixelScaleTag (33550)
        if 'ModelPixelScaleTag' in tags:
            scale = tags['ModelPixelScaleTag'].value
            cell_w = float(scale[0])
            cell_h = float(scale[1])
        elif 33550 in tags:
            scale = tags[33550].value
            cell_w = float(scale[0])
            cell_h = float(scale[1])

        # ModelTiepointTag (33922) — [i, j, k, x, y, z]
        if 'ModelTiepointTag' in tags:
            tp = tags['ModelTiepointTag'].value
            origin_x = float(tp[3])
            origin_y = float(tp[4])
        elif 33922 in tags:
            tp = tags[33922].value
            origin_x = float(tp[3])
            origin_y = float(tp[4])

        # ModelTransformationTag (34264) — 4x4 affine
        if cell_w == 1.0 and cell_h == 1.0:
            for tag_key in ('ModelTransformationTag', 34264):
                if tag_key in tags:
                    t = tags[tag_key].value
                    cell_w = abs(float(t[0]))
                    cell_h = abs(float(t[5]))
                    origin_x = float(t[3])
                    origin_y = float(t[7])
                    break

        if cell_w == 0:
            cell_w = 1.0
        if cell_h == 0:
            cell_h = 1.0

        logger.info(f"GeoTIFF geotransform: origin=({origin_x}, {origin_y}), pixel=({cell_w}, {cell_h})")
        return origin_x, origin_y, cell_w, cell_h

    def _get_nodata(self, tags) -> Optional[float]:
        """Extract nodata value from TIFF tags."""
        for key in ('GDAL_NODATA', 42113):
            if key in tags:
                try:
                    return float(tags[key].value)
                except (ValueError, TypeError):
                    pass
        return None

    # ── ASCII Grid (.asc) ────────────────────────────────────────────────

    def _parse_asc(self, path: Path, stride: int) -> pv.StructuredGrid:
        """Parse ESRI ASCII Grid format."""
        header = {}
        header_lines = 0

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0].lower() in (
                    'ncols', 'nrows', 'xllcorner', 'yllcorner',
                    'xllcenter', 'yllcenter', 'cellsize', 'nodata_value',
                    'dx', 'dy',
                ):
                    header[parts[0].lower()] = parts[1]
                    header_lines += 1
                else:
                    break

        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        cellsize = float(header.get('cellsize', header.get('dx', '1')))
        nodata = float(header.get('nodata_value', '-9999'))

        # Origin: corner vs center convention
        if 'xllcorner' in header:
            origin_x = float(header['xllcorner'])
            origin_y = float(header['yllcorner'])
        else:
            origin_x = float(header.get('xllcenter', '0')) - cellsize / 2
            origin_y = float(header.get('yllcenter', '0')) - cellsize / 2

        logger.info(f"ASC Grid: {ncols}x{nrows}, cellsize={cellsize}, origin=({origin_x}, {origin_y})")

        # Read data body
        data = np.loadtxt(path, skiprows=header_lines)
        if data.shape != (nrows, ncols):
            data = data.reshape(nrows, ncols)

        # Replace nodata with NaN
        data = data.astype(np.float64)
        data[np.isclose(data, nodata, rtol=1e-5)] = np.nan

        return self._build_structured_grid(data, origin_x, origin_y, cellsize, cellsize, stride)

    # ── Surfer Grid (.grd) ───────────────────────────────────────────────

    def _parse_grd(self, path: Path, stride: int) -> pv.StructuredGrid:
        """Parse Surfer 6 ASCII (DSAA) or Surfer 7 binary (DSRB) grid."""
        with open(path, 'rb') as f:
            magic = f.read(4)

        if magic == b'DSAA':
            return self._parse_grd_ascii(path, stride)
        elif magic == b'DSRB':
            return self._parse_grd_binary(path, stride)
        else:
            # Try ASCII anyway (some .grd files omit DSAA header)
            return self._parse_grd_ascii(path, stride)

    def _parse_grd_ascii(self, path: Path, stride: int) -> pv.StructuredGrid:
        """Parse Surfer 6 ASCII grid (DSAA format)."""
        with open(path, 'r') as f:
            lines = f.readlines()

        idx = 0
        # Skip DSAA header if present
        if lines[0].strip() == 'DSAA':
            idx = 1

        nx, ny = map(int, lines[idx].split())
        idx += 1
        xmin, xmax = map(float, lines[idx].split())
        idx += 1
        ymin, ymax = map(float, lines[idx].split())
        idx += 1
        zmin, zmax = map(float, lines[idx].split())
        idx += 1

        # Read remaining data
        values = []
        for line in lines[idx:]:
            values.extend(float(v) for v in line.split())

        data = np.array(values).reshape(ny, nx)
        cell_w = (xmax - xmin) / max(1, nx - 1)
        cell_h = (ymax - ymin) / max(1, ny - 1)

        # Surfer blanking value
        blank_val = 1.70141e+38
        data[data >= blank_val] = np.nan

        logger.info(f"Surfer ASCII Grid: {nx}x{ny}, x=[{xmin},{xmax}], y=[{ymin},{ymax}]")
        return self._build_structured_grid(data, xmin, ymin, cell_w, cell_h, stride)

    def _parse_grd_binary(self, path: Path, stride: int) -> pv.StructuredGrid:
        """Parse Surfer 7 binary grid (DSRB format)."""
        with open(path, 'rb') as f:
            f.read(4)  # 'DSRB'
            f.read(4)  # header size

            # Grid section
            section_id = f.read(4)
            section_size = struct.unpack('<i', f.read(4))[0]

            ny = struct.unpack('<i', f.read(4))[0]
            nx = struct.unpack('<i', f.read(4))[0]
            xmin = struct.unpack('<d', f.read(8))[0]
            ymin = struct.unpack('<d', f.read(8))[0]
            cell_w = struct.unpack('<d', f.read(8))[0]
            cell_h = struct.unpack('<d', f.read(8))[0]
            zmin = struct.unpack('<d', f.read(8))[0]
            zmax = struct.unpack('<d', f.read(8))[0]
            rotation = struct.unpack('<d', f.read(8))[0]
            blank_val = struct.unpack('<d', f.read(8))[0]

            # Data section
            section_id = f.read(4)
            section_size = struct.unpack('<i', f.read(4))[0]
            data = np.frombuffer(f.read(nx * ny * 8), dtype='<f8').reshape(ny, nx)

        data = data.copy()
        data[data >= blank_val] = np.nan

        logger.info(f"Surfer Binary Grid: {nx}x{ny}, cell=({cell_w},{cell_h})")
        return self._build_structured_grid(data, xmin, ymin, cell_w, cell_h, stride)

    # ── Common grid builder ──────────────────────────────────────────────

    def _build_structured_grid(
        self,
        elevation: np.ndarray,
        origin_x: float,
        origin_y: float,
        cell_w: float,
        cell_h: float,
        stride: int = 1,
    ) -> pv.StructuredGrid:
        """
        Build a PyVista StructuredGrid from a 2D elevation array.

        The grid is oriented so that row 0 = north (top of raster).
        """
        if stride > 1:
            elevation = elevation[::stride, ::stride]
            cell_w *= stride
            cell_h *= stride

        nrows, ncols = elevation.shape
        total_pts = nrows * ncols

        if total_pts > 20_000_000:
            # Auto-downsample extremely large rasters
            auto_stride = int(np.ceil(np.sqrt(total_pts / 4_000_000)))
            logger.warning(f"Raster too large ({total_pts:,} pts), auto-downsampling stride={auto_stride}")
            elevation = elevation[::auto_stride, ::auto_stride]
            cell_w *= auto_stride
            cell_h *= auto_stride
            nrows, ncols = elevation.shape

        # Build X, Y coordinate arrays
        x = origin_x + np.arange(ncols) * cell_w
        y = origin_y + np.arange(nrows) * cell_h
        # Flip Y so row 0 (north) maps to max Y
        y = y[::-1]

        xx, yy = np.meshgrid(x, y)

        # Replace NaN elevations with local mean for mesh continuity
        zz = elevation.copy()
        nan_mask = np.isnan(zz)
        if nan_mask.any():
            from scipy.ndimage import uniform_filter
            filled = np.where(nan_mask, 0.0, zz)
            count = np.where(nan_mask, 0.0, 1.0)
            filled_avg = uniform_filter(filled, size=5, mode='nearest')
            count_avg = uniform_filter(count, size=5, mode='nearest')
            count_avg[count_avg == 0] = 1
            zz[nan_mask] = filled_avg[nan_mask] / count_avg[nan_mask]

        grid = pv.StructuredGrid(xx, yy, zz)
        grid.point_data['Elevation'] = elevation.ravel(order='C')

        logger.info(f"Built StructuredGrid: {ncols}x{nrows} = {ncols * nrows:,} points")
        return grid
