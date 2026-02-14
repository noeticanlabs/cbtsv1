#!/usr/bin/env python3
"""
Lexicon Conformance Validator

Validates that documentation and code conform to the master lexicon defined in
docs/coherence/lexicon.json.

This script:
1. Loads the master lexicon
2. Checks for prohibited terms in code/documentation
3. Validates JSON schema of the lexicon itself
4. Verifies cross-repo consistency (if multiple paths provided)

Usage:
    python lexicon_validate.py [--verbose] [paths...]
    
Example:
    python lexicon_validate.py --verbose src/ docs/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple


class LexiconValidator:
    """Validates codebase against the master lexicon."""
    
    def __init__(self, lexicon_path: str, verbose: bool = False):
        self.verbose = verbose
        self.lexicon_path = Path(lexicon_path)
        self.lexicon = None
        self.errors = []
        self.warnings = []
        
    def load_lexicon(self) -> bool:
        """Load and parse the lexicon JSON."""
        try:
            with open(self.lexicon_path, 'r') as f:
                self.lexicon = json.load(f)
            
            # Validate basic structure
            required_fields = ['lexicon_version', 'canonical', 'definitions', 'prohibited_terms']
            for field in required_fields:
                if field not in self.lexicon:
                    self.errors.append(f"Lexicon missing required field: {field}")
                    return False
            
            if self.verbose:
                print(f"✓ Loaded lexicon v{self.lexicon.get('lexicon_version', 'unknown')}")
                print(f"  Definitions: {len(self.lexicon.get('definitions', {}))}")
                print(f"  Prohibited terms: {len(self.lexicon.get('prohibited_terms', []))}")
            
            return True
        except FileNotFoundError:
            self.errors.append(f"Lexicon not found: {self.lexicon_path}")
            return False
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in lexicon: {e}")
            return False
    
    def check_prohibited_terms(self, file_paths: List[str]) -> bool:
        """Check for prohibited terms in source files."""
        prohibited = set(self.lexicon.get('prohibited_terms', []))
        if not prohibited:
            return True
        
        found_prohibited = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue
            
            # If it's a directory, collect all relevant files
            if path.is_dir():
                files_to_check = []
                for ext in ['.py', '.md', '.txt', '.rst', '.nllc']:
                    files_to_check.extend(path.rglob(f'*{ext}'))
            else:
                files_to_check = [path]
            
            for file_path in files_to_check:
                # Skip certain file types
                if file_path.suffix in ['.json', '.yaml', '.yml', '.lock']:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for term in prohibited:
                        # Word boundary matching
                        pattern = r'\b' + re.escape(term) + r'\b'
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            found_prohibited.append((str(file_path), term, len(matches)))
                except Exception as e:
                    if self.verbose:
                        self.warnings.append(f"Could not read {file_path}: {e}")
        
        if found_prohibited:
            for file_path, term, count in found_prohibited:
                self.errors.append(
                    f"PROHIBITED TERM '{term}' found in {file_path} ({count} occurrences)"
                )
            return False
        
        if self.verbose and not found_prohibited:
            print(f"✓ No prohibited terms found in {len(file_paths)} files")
        
        return True
    
    def validate_lexicon_json(self) -> bool:
        """Validate the lexicon JSON structure."""
        if not self.lexicon:
            return False
        
        valid = True
        
        # Check definitions have required fields
        for term_name, term_def in self.lexicon.get('definitions', {}).items():
            if 'definition' not in term_def:
                self.errors.append(f"Definition '{term_name}' missing 'definition' field")
                valid = False
            if 'type' not in term_def:
                self.errors.append(f"Definition '{term_name}' missing 'type' field")
                valid = False
        
        # Check axioms
        axioms = self.lexicon.get('axioms', {})
        expected_axioms = ['c1_state_legibility', 'c2_constraint_primacy', 
                          'c3_conservation_of_debt', 'c4_temporal_accountability',
                          'c5_irreversibility_of_failure']
        for axiom in expected_axioms:
            if axiom not in axioms:
                self.warnings.append(f"Axiom '{axiom}' not found in lexicon")
        
        # Check deprecated files exist in deprecates list
        deprecates = self.lexicon.get('deprecates', [])
        for deprecated_path in deprecates:
            deprecated_full = Path(deprecated_path)
            # Check relative to repo root
            if not deprecated_full.exists():
                # Try relative to lexicon location
                alt_path = self.lexicon_path.parent.parent / deprecated_path
                if not alt_path.exists():
                    self.warnings.append(f"Deprecated file not found: {deprecated_path}")
        
        if self.verbose and valid:
            print(f"✓ Lexicon JSON structure valid")
        
        return valid
    
    def check_code_lexicon_alignment(self, code_dir: str) -> bool:
        """Check that key lexicon terms are referenced in code."""
        # Key terms that should exist in code
        required_terms = [
            'residual', 'scale', 'weight', 'coherence_functional',
            'receipt', 'ledger', 'gate'
        ]
        
        code_path = Path(code_dir)
        if not code_path.exists():
            if self.verbose:
                print(f"⚠ Code directory not found: {code_dir}")
            return True  # Skip if no code directory
        
        # Look for Python files
        py_files = list(code_path.rglob('*.py'))
        
        found_terms = set()
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                for term in required_terms:
                    if term in content.lower():
                        found_terms.add(term)
            except Exception:
                pass
        
        missing = set(required_terms) - found_terms
        if missing:
            self.warnings.append(
                f"Key lexicon terms not found in code: {', '.join(missing)}"
            )
        
        if self.verbose:
            print(f"✓ Found {len(found_terms)}/{len(required_terms)} key terms in code")
        
        return True  # Warnings only, not errors
    
    def validate(self, paths: List[str], code_dir: str = None) -> bool:
        """Run all validations."""
        print("=" * 60)
        print("LEXICON CONFORMANCE VALIDATOR")
        print("=" * 60)
        
        # Step 1: Load lexicon
        print(f"\n[1/4] Loading lexicon from {self.lexicon_path}...")
        if not self.load_lexicon():
            self.print_results()
            return False
        
        # Step 2: Validate lexicon JSON structure
        print("\n[2/4] Validating lexicon JSON structure...")
        if not self.validate_lexicon_json():
            self.print_results()
            return False
        
        # Step 3: Check for prohibited terms
        print("\n[3/4] Checking for prohibited terms...")
        if not self.check_prohibited_terms(paths):
            self.print_results()
            return False
        
        # Step 4: Check code alignment (optional)
        if code_dir:
            print("\n[4/4] Checking code-lexicon alignment...")
            self.check_code_lexicon_alignment(code_dir)
        
        self.print_results()
        return len(self.errors) == 0
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✓ All validations passed!")
        
        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate documentation and code against the master lexicon"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--lexicon', '-l',
        default='docs/framework/lexicon.json',
        help='Path to master lexicon JSON (default: docs/framework/lexicon.json)'
    )
    parser.add_argument(
        '--code-dir', '-c',
        default='src',
        help='Code directory to check for term alignment'
    )
    parser.add_argument(
        'paths',
        nargs='*',
        default=['docs/', 'src/'],
        help='Paths to check for prohibited terms (default: docs/ src/)'
    )
    
    args = parser.parse_args()
    
    # Resolve lexicon path - default is relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    lexicon_path = (repo_root / args.lexicon).resolve()
    
    validator = LexiconValidator(str(lexicon_path), verbose=args.verbose)
    
    success = validator.validate(args.paths, args.code_dir)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
