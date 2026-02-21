"""
Backup SGSIM Panel Before Replacing Visualization Code
"""
import shutil
from datetime import datetime
from pathlib import Path

# File to backup
source = Path("block_model_viewer/ui/sgsim_panel.py")

# Create backup with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = Path(f"block_model_viewer/ui/sgsim_panel_BACKUP_{timestamp}.py")

try:
    # Copy file
    shutil.copy2(source, backup)

    # Verify backup exists and has content
    if backup.exists() and backup.stat().st_size > 0:
        print(f"SUCCESS: Backup created")
        print(f"  {backup}")
        print(f"  Size: {backup.stat().st_size:,} bytes")
        print()
        print("You can now safely edit sgsim_panel.py")
        print()
        print("To restore the backup if needed:")
        print(f"  python restore_sgsim.py")
    else:
        print("ERROR: Backup failed - file is empty or missing")

except Exception as e:
    print(f"ERROR: Backup failed: {e}")
