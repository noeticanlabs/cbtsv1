#!/usr/bin/env python3
"""
Import Checker Script for cbtsv1 project.

Scans Python files in src/ directory to find import issues:
1. Missing 'src.' prefix for modules in src/
2. Wrong 'cbtsv1.' prefix that should be 'src.cbtsv1.'
3. Naked imports like 'nsc' that should be 'src.nsc'
"""

import os
import re
import sys
from pathlib import Path
from typing import Set, List, Tuple, Dict

# Project root is the parent of scripts/
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Known packages in src/
VALID_PACKAGES = {
    'aml', 'aeonic', 'common', 'contracts', 'core', 
    'elliptic', 'hadamard', 'module', 'nllc', 'nsc', 
    'phaseloom', 'receipts', 'solver', 'spectral', 
    'tgs', 'triaxis', 'cbtsv1'
}


def get_all_python_files(src_dir: Path) -> List[Path]:
    """Get all Python files in src/ directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files


def get_valid_modules(src_dir: Path) -> Set[str]:
    """Get all valid module names from src/ directory."""
    modules = set()
    
    for item in src_dir.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            # It's a package directory
            if (item / '__init__.py').exists():
                modules.add(item.name)
        elif item.suffix == '.py' and item.stem != '__init__':
            # It's a module file
            modules.add(item.stem)
    
    return modules


def find_imports_in_file(file_path: Path) -> List[Tuple[int, str]]:
    """Find all import statements in a Python file."""
    imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return imports
    
    for i, line in enumerate(lines, 1):
        # Match both "from X import Y" and "import X"
        # Strip comments
        line_stripped = line.split('#')[0].strip()
        
        if line_stripped.startswith('from '):
            match = re.match(r'from\s+([\w.]+)', line_stripped)
            if match:
                imports.append((i, match.group(1)))
        elif line_stripped.startswith('import '):
            match = re.match(r'import\s+([\w.]+)', line_stripped)
            if match:
                imports.append((i, match.group(1)))
    
    return imports


def check_imports(src_dir: Path) -> Dict[str, List[Dict]]:
    """
    Check all Python files for import issues.
    
    Returns a dictionary with categories:
    - 'missing_src_prefix': imports missing 'src.' prefix
    - 'wrong_cbtsv1_prefix': imports using 'cbtsv1.' instead of 'src.cbtsv1.'
    - 'naked_imports': bare module imports without any prefix
    """
    issues = {
        'missing_src_prefix': [],
        'wrong_cbtsv1_prefix': [],
        'naked_imports': []
    }
    
    valid_modules = get_valid_modules(src_dir)
    python_files = get_all_python_files(src_dir)
    
    print(f"Found {len(python_files)} Python files in {src_dir}")
    print(f"Valid modules in src/: {sorted(valid_modules)}")
    print("-" * 60)
    
    for file_path in python_files:
        rel_path = file_path.relative_to(src_dir)
        imports = find_imports_in_file(file_path)
        
        for line_num, module_path in imports:
            # Get the first component of the import
            first_component = module_path.split('.')[0]
            
            # Check for naked imports (bare module names that exist in src/)
            if first_component in VALID_PACKAGES and not module_path.startswith('src.'):
                # This is a naked import like 'nsc' or 'aml' or 'cbtsv1'
                # We need to be more specific - check if it's a relative import
                if not module_path.startswith('.'):
                    # Check if this is the first component (e.g., "from nsc import ...")
                    if module_path == first_component:
                        issues['naked_imports'].append({
                            'file': str(rel_path),
                            'line': line_num,
                            'import': module_path,
                            'suggestion': f'from src.{module_path} import ...'
                        })
                    elif not any(module_path.startswith(prefix) for prefix in ['src.', 'cbtsv1.']):
                        # This is like "from nsc.types import ..." without src prefix
                        issues['naked_imports'].append({
                            'file': str(rel_path),
                            'line': line_num,
                            'import': module_path,
                            'suggestion': f'from src.{module_path} import ...'
                        })
            
            # Check for wrong 'cbtsv1.' prefix
            if module_path.startswith('cbtsv1.'):
                issues['wrong_cbtsv1_prefix'].append({
                    'file': str(rel_path),
                    'line': line_num,
                    'import': module_path,
                    'suggestion': module_path.replace('cbtsv1.', 'src.cbtsv1.')
                })
            
            # Check for imports that should have 'src.' prefix
            # These are imports starting with a valid package name that is NOT src
            if first_component in VALID_PACKAGES and first_component != 'src':
                if not module_path.startswith('src.') and not module_path.startswith('.'):
                    # Make sure it's not already correct (e.g., src.aml)
                    if not any(module_path.startswith(p + '.') for p in ['src', 'cbtsv1']):
                        issues['missing_src_prefix'].append({
                            'file': str(rel_path),
                            'line': line_num,
                            'import': module_path,
                            'suggestion': f'from src.{module_path} import ...'
                        })
    
    return issues


def print_report(issues: Dict[str, List[Dict]]) -> None:
    """Print a formatted report of import issues."""
    total_issues = sum(len(v) for v in issues.values())
    
    print("\n" + "=" * 60)
    print("IMPORT CHECKER REPORT")
    print("=" * 60)
    
    # Wrong cbtsv1 prefix
    if issues['wrong_cbtsv1_prefix']:
        print(f"\n❌ WRONG 'cbtsv1.' PREFIX (should be 'src.cbtsv1.') [{len(issues['wrong_cbtsv1_prefix'])} issues]")
        print("-" * 60)
        for issue in issues['wrong_cbtsv1_prefix']:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Found:    {issue['import']}")
            print(f"    Should be: {issue['suggestion']}")
    
    # Missing src prefix
    if issues['missing_src_prefix']:
        print(f"\n❌ MISSING 'src.' PREFIX [{len(issues['missing_src_prefix'])} issues]")
        print("-" * 60)
        for issue in issues['missing_src_prefix']:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Found:    {issue['import']}")
            print(f"    Should be: {issue['suggestion']}")
    
    # Naked imports
    if issues['naked_imports']:
        print(f"\n❌ NAKED IMPORTS (missing any prefix) [{len(issues['naked_imports'])} issues]")
        print("-" * 60)
        for issue in issues['naked_imports']:
            print(f"  {issue['file']}:{issue['line']}")
            print(f"    Found:    {issue['import']}")
            print(f"    Should be: {issue['suggestion']}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_issues} total issues found")
    print("=" * 60)
    
    if total_issues == 0:
        print("✅ No import issues found!")
    else:
        print(f"  - Wrong 'cbtsv1.' prefix: {len(issues['wrong_cbtsv1_prefix'])}")
        print(f"  - Missing 'src.' prefix: {len(issues['missing_src_prefix'])}")
        print(f"  - Naked imports: {len(issues['naked_imports'])}")


def main():
    """Main entry point."""
    print("CBTSV1 Import Checker")
    print("=" * 60)
    
    if not SRC_DIR.exists():
        print(f"Error: src/ directory not found at {SRC_DIR}")
        sys.exit(1)
    
    issues = check_imports(SRC_DIR)
    print_report(issues)
    
    # Return exit code based on issues found
    total_issues = sum(len(v) for v in issues.values())
    return 1 if total_issues > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
