**GeoX — Scan Analysis & Fragmentation Pipeline Design**

Purpose: Describe scan data ingestion, validation, cleaning, segmentation, fragmentation analysis, and PSD computation for drone scans found under `scans/`.

---

## Overview

- **Location**: `block_model_viewer/scans/`
- **Responsibilities**: Point cloud/mesh import, validation, cleaning, fragmentation segmentation, metrics computation, PSD analysis, and audit trails.
- **Scope**: Production-grade scan analysis subsystem completely separate from drillholes, geostats, and block models.
- **Architecture**: Pipeline-based with explicit stages, provenance tracking, and deterministic algorithms.

## Key Pipeline Stages

### 1. Scan Ingest
- **Input**: Raw scan files (LAS/LAZ, PLY, OBJ, STL, XYZ, E57)
- **Process**: Format detection, parsing, CRS extraction, unit detection
- **Output**: `ScanData` object with points, normals, colors, metadata
- **Failure Modes**: Unsupported format, corrupted file, missing CRS

### 2. Scan Validation
- **Input**: `ScanData` object
- **Process**: Coordinate validation, density checks, outlier detection, data quality assessment
- **Output**: `ValidationReport` with explicit error/warning lists
- **Failure Modes**: Missing CRS, insufficient density, coordinate outliers, non-finite values

### 3. Scan Cleaning
- **Input**: Validated `ScanData`
- **Process**: Outlier removal, duplicate elimination, normal estimation
- **Output**: Cleaned `ScanData` with normals, `CleaningReport`
- **Failure Modes**: Normal estimation failure, excessive outliers

### 4. Fragment Segmentation
- **Input**: Cleaned `ScanData` with normals
- **Process**: Fragment identification using region growing or DBSCAN
- **Output**: Fragment labels array, `SegmentationResult`
- **Failure Modes**: Under-segmentation (single fragment), over-segmentation (too many small fragments)

### 5. Fragment Metrics & PSD
- **Input**: Segmented scan with fragment labels
- **Process**: Volume computation, shape analysis, confidence scoring, PSD percentiles
- **Output**: `FragmentMetrics` list, `PSDResults` with P10/P50/P80
- **Failure Modes**: Invalid fragment volumes, insufficient fragments for PSD

### 6. LOD Management (Performance)
- **Input**: Large point clouds
- **Process**: Octree spatial indexing, viewport culling, downsampling
- **Output**: Optimized point subsets for rendering
- **Failure Modes**: Memory exhaustion, performance degradation

## Public Interfaces

### Core Classes
```python
# Data models
ScanData              # Unified scan container
ValidationReport      # Validation results
CleaningReport        # Cleaning statistics
FragmentMetrics       # Individual fragment properties
PSDResults           # Particle size distribution

# Algorithms
ScanIngestor         # Format parsing
ScanValidator        # Data validation
ScanCleaner          # Outlier removal & normal estimation
RegionGrowingSegmenter  # Surface-based segmentation
DBSCANSegmenter      # Density-based segmentation
FragmentMetricsComputer  # Metrics computation
ScanLODController    # Performance optimization
```

### Controller Interface
```python
# Main pipeline orchestration
ScanController.load_scan(filepath) -> UUID
ScanController.validate_scan(scan_id) -> ValidationReport
ScanController.clean_scan(scan_id) -> UUID
ScanController.segment_scan(scan_id, strategy) -> Tuple[UUID, labels]
ScanController.compute_fragment_metrics(scan_id, labels) -> List[FragmentMetrics]
```

## Data Model

### ScanData Structure
```python
@dataclass
class ScanData:
    points: np.ndarray      # (N, 3) coordinates
    faces: Optional[np.ndarray]  # (M, 3) mesh faces
    normals: Optional[np.ndarray]  # (N, 3) surface normals
    colors: Optional[np.ndarray]  # (N, 3) RGB colors
    crs: Optional[str]      # Coordinate reference system
    units: str             # "meters" or "feet"
    file_format: str       # Source format
```

### FragmentMetrics Structure
```python
@dataclass
class FragmentMetrics:
    fragment_id: int
    point_count: int
    volume_m3: float
    equivalent_diameter_m: float
    sphericity: float          # 0-1 (perfect sphere = 1)
    elongation: float          # >= 1 (length/width)
    aspect_ratio: Tuple[float, float, float]  # (L, W, H)
    confidence_score: float    # 0-1 (data quality)
    centroid: Tuple[float, float, float]
```

## Supported File Formats

| Format | Extension | Support | Notes |
|--------|-----------|---------|-------|
| LAS/LAZ | .las, .laz | Full | CRS extraction, colors, intensities |
| PLY | .ply | Full | Binary/ASCII, meshes and point clouds |
| OBJ | .obj | Full | Mesh format with materials |
| STL | .stl | Full | Binary/ASCII mesh format |
| XYZ | .xyz | Full | Custom text format (X Y Z [R G B]) |
| E57 | .e57 | Planned | Complex format, not yet implemented |

## Audit & Provenance

### Provenance Chain
- **Every transformation** creates a new `ProcessingStep` entry
- **Parameters snapshotted** at execution time (deep copy)
- **Input/output versions** linked via UUIDs
- **Immutable history** - no mutation of past records
- **Checksums** computed for all intermediate outputs

### Registry Structure
```python
@dataclass
class ScanMetadata:
    scan_id: UUID                    # Immutable identifier
    source_file: Path               # Original file path
    source_hash: str               # SHA-256 checksum
    processing_history: List[ProcessingStep]  # Immutable chain
    derived_products: List[DerivedProduct]   # Fragment results, PSD
```

## Error Handling & Failure Modes

### Explicit Failures (Non-Silent)
All failures produce **explicit errors** or **warnings** - never silent fallback:

| Stage | Failure | Detection | Response |
|-------|---------|-----------|----------|
| **Ingest** | Missing CRS | Parser reports `crs=None` | Error: "CRS required for mining analysis" |
| **Ingest** | Corrupted file | Parse failure | Error: "File corrupted or unreadable" |
| **Validation** | Low density | Points/m³ < threshold | Error: "Insufficient density for analysis" |
| **Validation** | Outliers | Statistical detection | Warning: "Outliers detected, recommend cleaning" |
| **Segmentation** | Under-segmentation | Single fragment > 90% points | Error: "Segmentation failed - single fragment" |
| **Segmentation** | Over-segmentation | Small fragments > 50% total | Warning: "High number of small fragments" |
| **Metrics** | Invalid volume | Volume ≤ 0 | Warning: "Fragment volume invalid, skipping" |
| **Metrics** | No fragments | Zero valid fragments | Error: "No fragments for PSD computation" |

### Error Recovery
- **Validation errors**: Block progression, require user action
- **Warnings**: Allow progression but log in provenance
- **Partial failures**: Process valid fragments, skip invalid ones
- **Cancellation**: Support operation cancellation with partial results

## Performance & LOD Strategy

### Octree Spatial Indexing
- **Purpose**: Efficient viewport culling for large point clouds
- **Implementation**: Custom octree with configurable depth and point limits
- **Benefits**: Maintains interactivity with millions of points

### Rendering Modes
- **Preview Mode**: Downsampled data (1:10 to 1:100 ratio)
- **Analysis Mode**: Full resolution data (no downsampling)
- **Automatic switching**: No - user explicitly selects mode

### Point Budget Management
- **Viewport limit**: Configurable (default 1M points)
- **Frustum culling**: Camera-based visibility testing
- **Level-of-detail**: Distance-based point density reduction

## Segmentation Algorithms

### Region Growing Strategy
- **Algorithm**: Surface normal similarity + curvature seeding
- **Parameters**:
  - `normal_threshold_deg`: Maximum angle difference (default 30°)
  - `curvature_threshold`: Minimum curvature for seeds (default 0.01)
  - `k_neighbors`: Neighbors for normal estimation (default 15)
- **Strengths**: Respects surface boundaries, handles complex shapes
- **Limitations**: Requires good normal estimates, sensitive to noise

### DBSCAN Strategy
- **Algorithm**: Density-based spatial clustering
- **Parameters**:
  - `epsilon`: Distance threshold (meters, default 0.05)
  - `min_points`: Minimum cluster size (default 20)
  - `use_hdbscan`: Variable density clustering (default False)
- **Strengths**: Handles arbitrary shapes, automatic cluster detection
- **Limitations**: Sensitive to parameter tuning, struggles with varying density

## UI Workflow & States

### State Machine
```
EMPTY → FILE_LOADED → VALIDATED → CLEANED → SEGMENTED → METRICS_READY
         ↓             ↓           ↓          ↓
      ERROR        ERROR       ERROR      ERROR
```

### User Workflow
1. **Load Scan**: File selection with format auto-detection
2. **Validate**: Automatic data quality checks with detailed report
3. **Clean**: Optional outlier removal and normal estimation
4. **Segment**: Choose algorithm (Region Growing/DBSCAN) with parameters
5. **Analyze**: Compute fragment metrics and PSD
6. **Export**: Save results as CSV, meshes, or reports

### Panel Features
- **Progress reporting**: Real-time stage progress with cancellation
- **Parameter controls**: Algorithm-specific parameter adjustment
- **Results visualization**: Fragment coloring by size, shape, confidence
- **Export options**: CSV metrics, OBJ/STL fragments, PSD curves

## Third-Party Dependencies

### Required Libraries
- **open3d>=0.17.0**: Point cloud processing, normal estimation, outlier removal
- **laspy>=2.5.0**: LAS/LAZ file format support
- **trimesh>=3.20.0**: Mesh I/O and processing
- **scikit-learn>=1.3.0**: DBSCAN clustering algorithm
- **scipy>=1.11.0**: Spatial algorithms, statistical functions

### Optional Libraries
- **hdbscan>=0.8.33**: Hierarchical DBSCAN for variable density clustering

### Library Selection Criteria
- **Production-grade**: Mature, well-tested libraries
- **Deterministic**: Algorithms produce consistent results
- **Performance**: Efficient for large datasets (100K+ points)
- **License compatibility**: Compatible with GeoX licensing

## Testing Strategy

### Unit Tests
- **Coverage**: All pipeline stages with synthetic data
- **Validation**: Known fragment counts, volumes, PSD values
- **Edge cases**: Empty files, malformed data, extreme parameters
- **Performance**: Memory usage, execution time benchmarks

### Integration Tests
- **End-to-end**: File → validation → cleaning → segmentation → metrics
- **Format support**: All supported file formats
- **Error handling**: Comprehensive failure mode testing
- **UI integration**: Panel state transitions, progress reporting

### Test Data
- **Synthetic**: Known geometries, controlled noise levels
- **Real scans**: Anonymized mining site data (when available)
- **Edge cases**: Single points, collinear arrangements, extreme scales

## Security Considerations

### File Input Validation
- **Format verification**: Strict format checking before parsing
- **Size limits**: Maximum file size and point count limits
- **Memory protection**: Streaming parsing for large files
- **Path sanitization**: Safe file path handling

### Data Integrity
- **Checksum validation**: SHA-256 verification of all inputs
- **Provenance tracking**: Immutable audit trail of all operations
- **Parameter validation**: Strict bounds checking on all inputs
- **Output validation**: Sanity checks on computed results

## Roadmap & Extensions

### Planned Features
- **E57 support**: Complex format parsing for industrial scanners
- **Real-time streaming**: Progressive loading for massive datasets
- **GPU acceleration**: CUDA/OpenCL optimization for large scans
- **Advanced segmentation**: Machine learning-based fragment detection
- **Multi-scan registration**: Automatic alignment of multiple scans

### Integration Points
- **Block model comparison**: Fragment volumes vs. block sizes
- **Grade control**: Fragment analysis for ore/waste discrimination
- **Reporting**: Automated fragmentation reports with charts
- **Simulation**: Fragment size distribution in blasting models

---

## Implementation Status

✅ **Completed**
- Scan registry with provenance tracking
- Multi-format file ingestion (LAS, PLY, OBJ, STL, XYZ)
- Comprehensive validation with explicit failure modes
- Outlier removal and normal estimation
- Region growing and DBSCAN segmentation
- Fragment metrics computation (volume, shape, PSD)
- LOD controller with octree spatial indexing
- Controller orchestration and job registry integration
- State machine UI panel with progress reporting
- Menu integration and panel registration
- Unit tests for all pipeline stages
- Documentation and design specification

🔄 **In Progress**
- Performance optimization for 10M+ point scans
- Advanced visualization (fragment coloring schemes)
- Export functionality (CSV, OBJ/STL)

📋 **Planned**
- E57 format support
- GPU acceleration
- Machine learning segmentation options
- Multi-scan registration and merging

---

## Validation Checklist

### Separation & Isolation
- [x] `ScanRegistry` separate from `DataRegistry`
- [x] No imports from `drillholes/` or `geostats/` modules
- [x] Scan panel doesn't use drillhole/block model UI widgets

### Determinism & Auditability
- [x] All algorithms accept explicit parameters
- [x] Parameters snapshotted in provenance chain
- [x] Checksums computed for intermediate outputs
- [x] Immutable processing history

### Error Handling
- [x] All failure modes produce explicit errors/warnings
- [x] Missing CRS triggers user prompt
- [x] Insufficient density produces clear error
- [x] Segmentation failures detected and reported

### Performance & LOD
- [x] Octree spatial indexing implemented
- [x] Preview vs analysis modes supported
- [x] Configurable point budget management
- [x] Memory-efficient processing for large scans

### UI Contract
- [x] Scan menu in top-level menu bar
- [x] Dedicated scan panel with empty state
- [x] State machine UI with clear workflow steps
- [x] Progress reporting with cancellation support

### Data Formats
- [x] LAS/LAZ parsing with CRS extraction
- [x] PLY, OBJ, STL mesh support
- [x] XYZ custom format parsing
- [x] Format auto-detection from extensions

### Segmentation
- [x] Region growing with normal/cuvature constraints
- [x] DBSCAN with epsilon/min_points parameters
- [x] Both strategies handle under/over-segmentation
- [x] Parameters exposed in UI with validation

### Metrics & PSD
- [x] Convex hull volume computation
- [x] Sphericity, elongation, aspect ratio calculation
- [x] Explicit percentile computation (no interpolation)
- [x] Confidence scoring based on data quality

### Integration
- [x] ScanController integrates with AppController
- [x] Job registry tasks registered and functional
- [x] VisController supports scan rendering
- [x] PanelManager registration complete

### Testing
- [x] Unit tests for all pipeline stages
- [x] Synthetic data generation utilities
- [x] Validation of PSD computation accuracy
- [x] Error handling and failure mode tests

### Documentation
- [x] Complete design specification
- [x] API documentation for public interfaces
- [x] Failure mode documentation
- [x] Parameter tuning guidelines
