"""
Restore SGSIM Panel from Most Recent Backup
"""
import shutil
from pathlib import Path
import glob

# Find most recent backup
backups = glob.glob("block_model_viewer/ui/sgsim_panel_BACKUP_*.py")

if not backups:
    print("ERROR: No backups found!")
    print("  Looking for: block_model_viewer/ui/sgsim_panel_BACKUP_*.py")
    exit(1)

# Sort by timestamp (filename)
latest_backup = sorted(backups)[-1]
target = Path("block_model_viewer/ui/sgsim_panel.py")

print(f"Found backup: {latest_backup}")
response = input("Restore this backup? (yes/no): ")

if response.lower() in ['yes', 'y']:
    try:
        shutil.copy2(latest_backup, target)
        print(f"SUCCESS: Restored from backup")
        print(f"  {latest_backup} -> {target}")
    except Exception as e:
        print(f"ERROR: Restore failed: {e}")
else:
    print("Restore cancelled")
