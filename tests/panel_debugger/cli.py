"""Command-line interface for GeoX Panel Debugger"""
import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="GeoX Panel Debugger - Comprehensive Panel Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m panel_debugger                    # Run all tests
  python -m panel_debugger --category signals  # Run specific category
  python -m panel_debugger --panel KrigingPanel # Test specific panel
  python -m panel_debugger -m critical        # Run only critical tests
        """
    )
    
    parser.add_argument('--category', 
                       choices=['panel_init', 'signals', 'data_flow', 'renderer', 'coordinates', 'performance', 'integration', 'all'],
                       default='all',
                       help='Test category to run (default: all)')
    
    parser.add_argument('--panel',
                       help='Test specific panel by class name')
    
    parser.add_argument('-m', '--marker',
                       help='Run tests with specific pytest marker (e.g., critical)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    parser.add_argument('--failfast', '-x',
                       action='store_true',
                       help='Stop on first failure')
    
    parser.add_argument('--html',
                       help='Generate HTML report at specified path')
    
    args = parser.parse_args()
    
    # Build pytest command
    pytest_args = [sys.executable, '-m', 'pytest']
    
    # Add test directory
    test_dir = Path(__file__).parent / 'tests'
    
    # Add category filter
    if args.category != 'all':
        pytest_args.extend(['-m', args.category])
    elif args.marker:
        pytest_args.extend(['-m', args.marker])
    
    # Add verbosity
    if args.verbose:
        pytest_args.append('-v')
    
    # Add failfast
    if args.failfast:
        pytest_args.append('-x')
    
    # Add HTML report
    if args.html:
        pytest_args.extend(['--html', args.html, '--self-contained-html'])
    
    pytest_args.append(str(test_dir))
    
    # Print header
    print("╔" + "═"*68 + "╗")
    print("║" + "  GeoX Panel Debugger v1.0".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # Run pytest
    return subprocess.call(pytest_args)

if __name__ == '__main__':
    sys.exit(main())
