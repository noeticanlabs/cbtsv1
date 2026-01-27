#!/usr/bin/env python3
"""
Script to update imports from gr_solver.* to src.core.*
This handles the migration from the old gr_solver/ directory to src/core/
"""

import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Core modules
    "from src.core.gr_solver import": "from src.core.gr_solver import",
    "from src.core.gr_stepper import": "from src.core.gr_stepper import",
    "from src.core.gr_constraints import": "from src.core.gr_constraints import",
    "from src.core.gr_geometry import": "from src.core.gr_geometry import",
    "from src.core.gr_geometry_nsc import": "from src.core.gr_geometry_nsc import",
    "from src.core.gr_gauge import": "from src.core.gr_gauge import",
    "from src.core.gr_scheduler import": "from src.core.gr_scheduler import",
    "from src.core.gr_ledger import": "from src.core.gr_ledger import",
    "from src.core.gr_clock import": "from src.core.gr_clock import",
    "from src.core.gr_clocks import": "from src.core.gr_clocks import",
    "from src.core.gr_core_fields import": "from src.core.gr_core_fields import",
    "from src.core.gr_rhs import": "from src.core.gr_rhs import",
    "from src.core.gr_loc import": "from src.core.gr_loc import",
    "from src.core.gr_sem import": "from src.core.gr_sem import",
    "from src.core.gr_gates import": "from src.core.gr_gates import",
    "from src.core.gr_ttl_calculator import": "from src.core.gr_ttl_calculator import",
    "from src.core.gr_coherence import": "from src.core.gr_coherence import",
    "from src.core.gr_receipts import": "from src.core.gr_receipts import",
    
    # Moved modules
    "from src.phaseloom.phaseloom_memory import": "from src.phaseloom.phaseloom_memory import",
    "from src.phaseloom.phaseloom_rails_gr import": "from src.phaseloom.phaseloom_rails_gr import",
    "from src.spectral.cache import": "from src.spectral.cache import",
    "from src.elliptic.solver import": "from src.elliptic.solver import",
    
    # Host API
    "from src.host_api import": "from src.host_api import",
}

def update_imports(file_path: Path) -> bool:
    """Update import statements in a single file."""
    try:
        content = file_path.read_text()
        original = content
        
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        if content != original:
            file_path.write_text(content)
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return False

def main():
    """Main function to update all Python files."""
    updated_files = []
    
    for py_file in Path(".").rglob("*.py"):
        # Skip certain directories
        if any(part in py_file.parts for part in [".git", ".pytest_cache", "__pycache__", "src", "gr_solver"]):
            continue
        if py_file.parent.name == "src":
            continue
            
        if update_imports(py_file):
            updated_files.append(py_file)
            print(f"Updated: {py_file}")
    
    print(f"\nTotal files updated: {len(updated_files)}")

if __name__ == "__main__":
    main()
