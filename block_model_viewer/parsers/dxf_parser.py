"""
DXF parser for importing CAD topographic surfaces, contours, and 3D geometry.

Extracts 3DFACE entities, polylines, meshes, and points from DXF files
and converts them to PyVista objects for 3D visualization in GeoX.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ['.dxf']


class DXFParser:
    """Parser for DXF CAD files -> PyVista PolyData surfaces and polylines."""

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path, layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse a DXF file into PyVista geometry.

        Args:
            file_path: Path to the .dxf file.
            layers: Optional list of DXF layer names to import.
                    If None, imports all layers.

        Returns:
            dict with keys:
                'surfaces': list[pv.PolyData]  — triangulated surfaces from 3DFACE / MESH
                'lines': list[pv.PolyData]     — polylines and contour lines
                'points': pv.PolyData | None   — point entities
                'layer_names': list[str]        — names of all layers found
                'summary': str                  — human-readable summary
        """
        try:
            import ezdxf
        except ImportError:
            raise ImportError(
                "ezdxf is required for DXF import. Install with: pip install ezdxf"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = ezdxf.readfile(str(file_path))
        msp = doc.modelspace()

        # Collect all layer names
        all_layers = sorted({e.dxf.layer for e in msp if hasattr(e.dxf, 'layer')})
        logger.info(f"DXF layers found: {all_layers}")

        # Filter entities by layer if requested
        if layers is not None:
            layer_set = set(layers)
            entities = [e for e in msp if hasattr(e.dxf, 'layer') and e.dxf.layer in layer_set]
        else:
            entities = list(msp)

        surfaces = self._extract_surfaces(entities)
        lines = self._extract_lines(entities)
        points = self._extract_points(entities)

        n_surf = sum(s.n_cells for s in surfaces) if surfaces else 0
        n_lines = len(lines)
        n_pts = points.n_points if points is not None else 0

        summary = (
            f"DXF import: {n_surf} surface faces, "
            f"{n_lines} polylines, {n_pts} points "
            f"from {len(all_layers)} layers"
        )
        logger.info(summary)

        return {
            'surfaces': surfaces,
            'lines': lines,
            'points': points,
            'layer_names': all_layers,
            'summary': summary,
        }

    # ── Surface extraction (3DFACE, MESH, POLYFACE) ──────────────────────

    def _extract_surfaces(self, entities) -> List[pv.PolyData]:
        """Extract triangulated surfaces from 3DFACE and MESH entities."""
        surfaces = []

        # 3DFACE entities
        faces_3d = [e for e in entities if e.dxftype() == '3DFACE']
        if faces_3d:
            surf = self._faces_to_polydata(faces_3d)
            if surf is not None:
                surfaces.append(surf)

        # MESH entities (ezdxf native mesh)
        meshes = [e for e in entities if e.dxftype() == 'MESH']
        for mesh_entity in meshes:
            surf = self._mesh_entity_to_polydata(mesh_entity)
            if surf is not None:
                surfaces.append(surf)

        # POLYFACE meshes (legacy DXF polymeshes)
        polyfaces = [
            e for e in entities
            if e.dxftype() == 'POLYLINE' and hasattr(e, 'is_poly_face_mesh')
            and e.is_poly_face_mesh
        ]
        for pf in polyfaces:
            surf = self._polyface_to_polydata(pf)
            if surf is not None:
                surfaces.append(surf)

        return surfaces

    def _faces_to_polydata(self, face_entities) -> Optional[pv.PolyData]:
        """Convert a list of 3DFACE entities to a single merged PolyData."""
        all_verts = []
        all_faces = []
        vert_offset = 0

        for face in face_entities:
            try:
                corners = [
                    np.array(face.dxf.vtx0),
                    np.array(face.dxf.vtx1),
                    np.array(face.dxf.vtx2),
                ]
                # 3DFACE can be triangle (vtx3 == vtx2) or quad
                vtx3 = np.array(face.dxf.vtx3)
                is_quad = not np.allclose(vtx3, corners[2])

                if is_quad:
                    corners.append(vtx3)

                for v in corners:
                    all_verts.append(v[:3])  # ensure XYZ only

                n = len(corners)
                face_conn = [n] + list(range(vert_offset, vert_offset + n))
                all_faces.extend(face_conn)
                vert_offset += n
            except Exception as e:
                logger.debug(f"Skipping malformed 3DFACE: {e}")

        if not all_verts:
            return None

        verts = np.array(all_verts, dtype=np.float64)
        faces = np.array(all_faces, dtype=np.int64)

        mesh = pv.PolyData(verts, faces=faces)
        logger.info(f"Extracted {len(face_entities)} 3DFACE entities -> {mesh.n_cells} faces")
        return mesh

    def _mesh_entity_to_polydata(self, mesh_entity) -> Optional[pv.PolyData]:
        """Convert an ezdxf MESH entity to PyVista PolyData."""
        try:
            vertices = np.array([list(v) for v in mesh_entity.vertices], dtype=np.float64)
            if len(vertices) == 0:
                return None

            face_list = []
            for face in mesh_entity.faces:
                indices = list(face)
                face_list.extend([len(indices)] + indices)

            if not face_list:
                # No faces — return as point cloud
                return pv.PolyData(vertices)

            faces = np.array(face_list, dtype=np.int64)
            mesh = pv.PolyData(vertices, faces=faces)
            logger.info(f"Extracted MESH entity: {mesh.n_points} vertices, {mesh.n_cells} faces")
            return mesh
        except Exception as e:
            logger.warning(f"Failed to extract MESH entity: {e}")
            return None

    def _polyface_to_polydata(self, polyface) -> Optional[pv.PolyData]:
        """Convert a POLYFACE mesh entity to PyVista PolyData."""
        try:
            from ezdxf.entities import DXFVertex

            vertices = []
            face_indices = []

            for item in polyface.vertices:
                if item.is_poly_face_mesh_vertex:
                    vertices.append(list(item.dxf.location)[:3])
                elif item.is_face_record:
                    # Face record stores 1-based vertex indices in vtx0..vtx3
                    idx = []
                    for attr in ('vtx0', 'vtx1', 'vtx2', 'vtx3'):
                        vi = getattr(item.dxf, attr, 0)
                        if vi != 0:
                            idx.append(abs(vi) - 1)  # convert 1-based to 0-based
                    if len(idx) >= 3:
                        face_indices.append(idx)

            if not vertices:
                return None

            verts = np.array(vertices, dtype=np.float64)
            face_list = []
            for fi in face_indices:
                face_list.extend([len(fi)] + fi)

            if face_list:
                mesh = pv.PolyData(verts, faces=np.array(face_list, dtype=np.int64))
            else:
                mesh = pv.PolyData(verts)

            logger.info(f"Extracted POLYFACE: {mesh.n_points} vertices, {mesh.n_cells} faces")
            return mesh
        except Exception as e:
            logger.warning(f"Failed to extract POLYFACE: {e}")
            return None

    # ── Line extraction (POLYLINE, LWPOLYLINE, LINE, SPLINE) ────────────

    def _extract_lines(self, entities) -> List[pv.PolyData]:
        """Extract polylines, lines, and splines as PyVista line meshes."""
        lines = []

        # LINE entities
        line_ents = [e for e in entities if e.dxftype() == 'LINE']
        for ent in line_ents:
            try:
                start = np.array(ent.dxf.start)[:3]
                end = np.array(ent.dxf.end)[:3]
                pts = np.vstack([start, end])
                conn = np.array([2, 0, 1], dtype=np.int64)
                lines.append(pv.PolyData(pts, lines=conn))
            except Exception:
                pass

        # LWPOLYLINE entities (2D with optional elevation)
        lw_ents = [e for e in entities if e.dxftype() == 'LWPOLYLINE']
        for ent in lw_ents:
            try:
                pts_2d = list(ent.get_points(format='xy'))
                if len(pts_2d) < 2:
                    continue
                elev = getattr(ent.dxf, 'elevation', 0.0)
                pts = np.array([[p[0], p[1], elev] for p in pts_2d], dtype=np.float64)
                n = len(pts)
                conn = [n] + list(range(n))
                polyline = pv.PolyData(pts, lines=np.array(conn, dtype=np.int64))
                lines.append(polyline)
            except Exception as e:
                logger.debug(f"Skipping LWPOLYLINE: {e}")

        # 3D POLYLINE entities
        poly_ents = [
            e for e in entities
            if e.dxftype() == 'POLYLINE'
            and hasattr(e, 'is_3d_polyline') and e.is_3d_polyline
        ]
        for ent in poly_ents:
            try:
                pts = np.array([list(v.dxf.location)[:3] for v in ent.vertices], dtype=np.float64)
                if len(pts) < 2:
                    continue
                n = len(pts)
                conn = [n] + list(range(n))
                polyline = pv.PolyData(pts, lines=np.array(conn, dtype=np.int64))
                lines.append(polyline)
            except Exception as e:
                logger.debug(f"Skipping 3D POLYLINE: {e}")

        # SPLINE entities — sample as polyline
        spline_ents = [e for e in entities if e.dxftype() == 'SPLINE']
        for ent in spline_ents:
            try:
                ctrl_pts = list(ent.control_points)
                if len(ctrl_pts) < 2:
                    continue
                pts = np.array([list(p)[:3] for p in ctrl_pts], dtype=np.float64)
                n = len(pts)
                conn = [n] + list(range(n))
                polyline = pv.PolyData(pts, lines=np.array(conn, dtype=np.int64))
                lines.append(polyline)
            except Exception as e:
                logger.debug(f"Skipping SPLINE: {e}")

        logger.info(f"Extracted {len(lines)} line/polyline entities")
        return lines

    # ── Point extraction ─────────────────────────────────────────────────

    def _extract_points(self, entities) -> Optional[pv.PolyData]:
        """Extract POINT entities as a PyVista point cloud."""
        point_ents = [e for e in entities if e.dxftype() == 'POINT']
        if not point_ents:
            return None

        pts = []
        for ent in point_ents:
            try:
                pts.append(list(ent.dxf.location)[:3])
            except Exception:
                pass

        if not pts:
            return None

        cloud = pv.PolyData(np.array(pts, dtype=np.float64))
        logger.info(f"Extracted {cloud.n_points} POINT entities")
        return cloud
