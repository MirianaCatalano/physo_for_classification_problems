from pathlib import Path
import shutil
import physo


# =========================
# PROJECT DIRECTORY
# =========================
project_dir = Path(__file__).resolve().parent.parent
modified_dimensional_analysis = project_dir / "physo_modifications" / "dimensional_analysis.py"
modified_learn = project_dir / "physo_modifications" / "learn.py"


# =========================
# FIND INSTALLED PHYOSO FILE
# =========================
physo_dir = Path(physo.__file__).resolve().parent
target_dimensional_analysis = physo_dir / "physym" / "dimensional_analysis.py"
target_learn = physo_dir / "learn" / "learn.py"


# =========================
# CHECK FILES
# =========================
if not modified_dimensional_analysis.exists():
    raise FileNotFoundError(
        f"Modified PhySO file not found: {modified_dimensional_analysis}"
    )
    
if not modified_learn.exists():
    raise FileNotFoundError(
        f"Modified PhySO file not found: {modified_learn}"
    )

if not target_dimensional_analysis.exists():
    raise FileNotFoundError(
        f"Installed PhySO file not found: {target_dimensional_analysis}"
    )

if not target_learn.exists():
    raise FileNotFoundError(
        f"Installed PhySO file not found: {target_learn}"
    )


# =========================
# BACKUP ORIGINAL FILE
# =========================
backup_dimensional_analysis = target_dimensional_analysis.with_suffix(".py.original")
backup_learn = target_learn.with_suffix(".py.original")

if not backup_dimensional_analysis.exists():
    shutil.copy2(target_dimensional_analysis, backup_dimensional_analysis)

if not backup_learn.exists():
    shutil.copy2(target_learn, backup_learn)


# =========================
# APPLY MODIFICATION
# =========================
shutil.copy2(modified_dimensional_analysis, target_dimensional_analysis)
shutil.copy2(modified_learn, target_learn)

print("PhySO modification applied successfully.")
print(f"Source: {modified_dimensional_analysis}")
print(f"Target: {target_dimensional_analysis}")
print(f"Source: {modified_learn}")
print(f"Target: {target_learn}")