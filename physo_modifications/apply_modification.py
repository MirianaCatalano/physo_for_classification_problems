from pathlib import Path
import shutil
import physo


# =========================
# PROJECT DIRECTORY
# =========================
project_dir = Path(__file__).resolve().parent.parent
modified_file = project_dir / "physo_modifications" / "dimensional_analysis.py"


# =========================
# FIND INSTALLED PHYOSO FILE
# =========================
physo_dir = Path(physo.__file__).resolve().parent
target_file = physo_dir / "physym" / "dimensional_analysis.py"


# =========================
# CHECK FILES
# =========================
if not modified_file.exists():
    raise FileNotFoundError(
        f"Modified PhySO file not found: {modified_file}"
    )

if not target_file.exists():
    raise FileNotFoundError(
        f"Installed PhySO file not found: {target_file}"
    )


# =========================
# BACKUP ORIGINAL FILE
# =========================
backup_file = target_file.with_suffix(".py.original")

if not backup_file.exists():
    shutil.copy2(target_file, backup_file)


# =========================
# APPLY MODIFICATION
# =========================
shutil.copy2(modified_file, target_file)

print("PhySO modification applied successfully.")
print(f"Source: {modified_file}")
print(f"Target: {target_file}")