# File Parsers

from .base_parser import BaseParser, ParserRegistry, parser_registry
from .csv_parser import CSVParser
from .vtk_parser import VTKParser
from .mesh_parser import MeshParser
from .mining_parser import MiningParser
from .structural_csv_parser import (
    StructuralCSVParser,
    CSVFormat,
    ColumnMapping,
    ParseResult,
    parse_structural_csv,
    detect_structural_csv_format,
)

# Register all parsers
parser_registry.register_parser(CSVParser())
parser_registry.register_parser(VTKParser())
parser_registry.register_parser(MeshParser())
parser_registry.register_parser(MiningParser())

__all__ = [
    'BaseParser',
    'ParserRegistry',
    'parser_registry',
    'CSVParser',
    'VTKParser',
    'MeshParser',
    'MiningParser',
    # Structural CSV parser
    'StructuralCSVParser',
    'CSVFormat',
    'ColumnMapping',
    'ParseResult',
    'parse_structural_csv',
    'detect_structural_csv_format',
]
