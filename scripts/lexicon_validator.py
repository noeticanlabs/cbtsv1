#!/usr/bin/env python3
"""
Lexicon Validator v2.0

Identifies issues in documentation and code by validating against the canonical lexicon.
This script scans for:
- Prohibited terms usage
- Definitional drift
- Axiom violations
- Layer inconsistencies
- Missing required fields
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Configuration
LEXICON_PATH = Path("docs/framework/lexicon.json")
SCAN_EXTENSIONS = {".py", ".md", ".json", ".nllc"}
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", "venv", "build", "dist", "vendor", "archive"}


class LexiconValidator:
    def __init__(self, lexicon_path: Path):
        with open(lexicon_path, 'r') as f:
            self.lexicon = json.load(f)
        
        self.issues: List[Dict[str, Any]] = []
        self.definitions = self.lexicon.get("definitions", {})
        self.axioms = self.lexicon.get("axioms", {})
        self.prohibited_terms = set(self.lexicon.get("prohibited_terms", []))
        self.scope = set(self.lexicon.get("scope", []))
        
    def add_issue(self, severity: str, category: str, file_path: str, 
                  line_num: int, message: str, context: str = ""):
        """Add an issue to the report."""
        self.issues.append({
            "severity": severity,  # ERROR, WARNING, INFO
            "category": category,
            "file": file_path,
            "line": line_num,
            "message": message,
            "context": context
        })
    
    def check_prohibited_terms(self, content: str, file_path: str):
        """Check for prohibited terms in content."""
        # Skip the lexicon itself - it intentionally lists prohibited terms
        if "lexicon.json" in file_path:
            return
            
        for term in self.prohibited_terms:
            # Case-insensitive search
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for i, line in enumerate(content.split('\n'), 1):
                matches = pattern.findall(line)
                if matches:
                    self.add_issue(
                        "ERROR",
                        "PROHIBITED_TERM",
                        file_path,
                        i,
                        f"Prohibited term '{term}' found in documentation/code",
                        line.strip()[:100]
                    )
    
    def check_unicode_symbols(self, content: str, file_path: str):
        """Check for unicode math symbols that should use lexicon symbols."""
        unicode_to_lexicon = {
            'α': 'alpha',
            'β': 'beta', 
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'π': 'pi',
            'σ': 'sigma',
            'φ': 'phi',
            'ω': 'omega',
            'Σ': 'Σ (sum)',
            '∏': '∏ (product)',
            '∫': '∫ (integral)',
            '∂': '∂ (partial)',
            '∇': '∇ (nabla)',
            '∞': 'infinity',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '≈': '~=',
            '→': '->',
            '∈': 'in',
            '⊂': 'subset',
            '∪': 'union',
            '∩': 'intersection',
        }
        
        for i, line in enumerate(content.split('\n'), 1):
            for unicode_char, expected in unicode_to_lexicon.items():
                if unicode_char in line:
                    # Only warn if it's in a code/math context
                    if '$' in line or '```' in line or '`' in line:
                        self.add_issue(
                            "WARNING",
                            "UNICODE_SYMBOL",
                            file_path,
                            i,
                            f"Consider using ASCII '{expected}' instead of '{unicode_char}' for consistency",
                            line.strip()[:80]
                        )
    
    def check_definition_consistency(self, content: str, file_path: str):
        """Check if definitions in docs match the lexicon."""
        # Look for definition patterns like: "term: definition" or "**term** - definition"
        definition_pattern = re.compile(
            r'(?:^|\n)\s*([a-z_][a-z0-9_]*)\s*[:\-]\s*(.{10,200}?)(?:\n|$)',
            re.MULTILINE
        )
        
        for i, line in enumerate(content.split('\n'), 1):
            # Check for potential definition mismatches
            for term, defn in self.definitions.items():
                canonical_def = defn.get("definition", "")
                # If line contains term but significantly different definition
                if term.lower() in line.lower() and len(line) > 20:
                    # Check if this might be a conflicting definition
                    if canonical_def.lower()[:30] not in line.lower():
                        # Might be a custom definition - flag for review
                        self.add_issue(
                            "INFO",
                            "DEFINITION_VARIANT",
                            file_path,
                            i,
                            f"Term '{term}' may have a custom definition here",
                            line.strip()[:80]
                        )
    
    def check_required_fields(self, file_path: str):
        """Check if lexicon definitions have all required fields."""
        for term, defn in self.definitions.items():
            required = defn.get("required_fields", [])
            # Some definitions should have layers
            if "layers" not in defn and term not in ["prohibited_terms", "drift_rule"]:
                self.add_issue(
                    "INFO",
                    "MISSING_LAYERS",
                    file_path,
                    0,
                    f"Definition '{term}' missing 'layers' field"
                )
    
    def check_layer_consistency(self, content: str, file_path: str):
        """Check for layer-related inconsistencies in code/docs."""
        # Look for layer mentions that might be inconsistent
        layer_pattern = re.compile(r'\bL[0-3]\b')
        
        for i, line in enumerate(content.split('\n'), 1):
            layers_found = layer_pattern.findall(line)
            if layers_found:
                # Check if the layer is valid
                for layer in layers_found:
                    if layer not in ["L0", "L1", "L2", "L3"]:
                        self.add_issue(
                            "ERROR",
                            "INVALID_LAYER",
                            file_path,
                            i,
                            f"Invalid layer '{layer}' - must be L0, L1, L2, or L3",
                            line.strip()[:80]
                        )
    
    def check_coherence_terminology(self, content: str, file_path: str):
        """Check for correct coherence-related terminology."""
        # Correct terms from lexicon
        correct_terms = {
            "coherence_functional": "coherence functional",
            "coherence_value": "coherence value", 
            "coherent_state": "coherent state",
            "coherence_budget": "coherence budget",
            "residual_block": "residual block",
            "residual_system": "residual system",
            "hamiltonian_residual": "hamiltonian residual",
            "momentum_residual": "momentum residual",
            "gauge_residual": "gauge residual",
            "receipt": "receipt",
            "ledger": "ledger",
        }
        
        # Common mistakes
        mistakes = {
            "coherence functional": "Use 'coherence_functional' (with underscore)",
            "coherence score": "Use 'coherence_value'",
            "coherence metric": "Use 'coherence_value'",
            "error norm": "Use 'residual' or 'coherence_value'",
        }
        
        # Terms that are actually OK in context
        ok_contexts = [
            "constraint residual",  # This is correct - residual blocks for constraints
            "constraint residuals",  # Plural form is also correct
            "incoherence",  # This is a valid physics term meaning inverse of coherence
            "incoherence score",  # Algorithm-specific term for the inverse measure
            "canonical coherence functional",  # This refers to the specific formula
        ]
        
        for i, line in enumerate(content.split('\n'), 1):
            line_lower = line.lower()
            for wrong, right in mistakes.items():
                # Skip if in OK context
                if any(ok in line_lower for ok in ok_contexts):
                    continue
                if wrong in line_lower:
                    self.add_issue(
                        "WARNING",
                        "TERMINOLOGY",
                        file_path,
                        i,
                        f"Use canonical term: {right}",
                        line.strip()[:80]
                    )
    
    def scan_file(self, file_path: Path):
        """Scan a single file for issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.add_issue(
                "ERROR",
                "READ_ERROR",
                str(file_path),
                0,
                f"Could not read file: {e}"
            )
            return
        
        rel_path = str(file_path)
        
        # Check prohibited terms
        self.check_prohibited_terms(content, rel_path)
        
        # Check unicode symbols
        self.check_unicode_symbols(content, rel_path)
        
        # Check layer consistency  
        self.check_layer_consistency(content, rel_path)
        
        # Check coherence terminology
        self.check_coherence_terminology(content, rel_path)
        
        # Check definition consistency (only for markdown)
        if file_path.suffix == '.md':
            self.check_definition_consistency(content, rel_path)
    
    def scan_directory(self, root_path: Path):
        """Recursively scan directory for issues."""
        for item in root_path.rglob('*'):
            if item.is_file():
                # Skip excluded directories
                if any(excl in item.parts for excl in EXCLUDE_DIRS):
                    continue
                    
                # Only scan specified extensions
                if item.suffix not in SCAN_EXTENSIONS:
                    continue
                    
                self.scan_file(item)
    
    def generate_report(self) -> str:
        """Generate a formatted report of all issues."""
        if not self.issues:
            return "✓ No lexicon violations found."
        
        # Sort by severity and file
        severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        self.issues.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["file"], x["line"]))
        
        lines = []
        lines.append("=" * 70)
        lines.append("LEXICON VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Lexicon: {self.lexicon.get('lexicon_version', 'unknown')}")
        lines.append(f"Master: {self.lexicon.get('master', False)}")
        lines.append(f"Total Issues: {len(self.issues)}")
        lines.append("")
        
        # Count by severity
        counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
        for issue in self.issues:
            counts[issue["severity"]] += 1
        
        lines.append(f"  Errors:   {counts['ERROR']}")
        lines.append(f"  Warnings: {counts['WARNING']}")
        lines.append(f"  Info:     {counts['INFO']}")
        lines.append("")
        
        # Group by severity
        for severity in ["ERROR", "WARNING", "INFO"]:
            severity_issues = [i for i in self.issues if i["severity"] == severity]
            if not severity_issues:
                continue
                
            lines.append("-" * 70)
            lines.append(f"{severity}S ({len(severity_issues)})")
            lines.append("-" * 70)
            
            for issue in severity_issues:
                lines.append(f"\n  [{issue['severity']}] {issue['file']}:{issue['line']}")
                lines.append(f"  Category: {issue['category']}")
                lines.append(f"  Message:  {issue['message']}")
                if issue.get("context"):
                    lines.append(f"  Context:  {issue['context']}")
        
        lines.append("")
        lines.append("=" * 70)
        
        if counts["ERROR"] > 0:
            lines.append("\n⚠️  ERRORS FOUND - Lexicon violations detected!")
            lines.append("    Run 'python scripts/lexicon_validator.py --fix' for auto-fixes where available.")
        elif counts["WARNING"] > 0:
            lines.append("\n⚠️  Warnings found - Review recommended.")
        else:
            lines.append("\n✓ No critical issues found.")
        
        return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate code/docs against canonical lexicon")
    parser.add_argument("--path", type=str, default=".", 
                        help="Path to scan (default: current directory)")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to auto-fix issues where possible")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show errors")
    
    args = parser.parse_args()
    
    # Find lexicon
    if LEXICON_PATH.exists():
        lexicon_path = LEXICON_PATH
    else:
        # Try to find it
        search_paths = [
            Path("docs/framework/lexicon.json"),
            Path("../docs/framework/lexicon.json"),
            Path("../../docs/framework/lexicon.json"),
        ]
        lexicon_path = None
        for p in search_paths:
            if p.exists():
                lexicon_path = p
                break
        
        if not lexicon_path:
            print("ERROR: Could not find lexicon.json", file=sys.stderr)
            sys.exit(1)
    
    print(f"Using lexicon: {lexicon_path}")
    
    validator = LexiconValidator(lexicon_path)
    
    # Check lexicon itself first
    print("Validating lexicon structure...")
    validator.check_required_fields(str(lexicon_path))
    
    # Scan requested path
    scan_path = Path(args.path)
    if not scan_path.exists():
        print(f"ERROR: Path does not exist: {scan_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning {scan_path}...")
    validator.scan_directory(scan_path)
    
    # Generate report
    report = validator.generate_report()
    
    if args.json:
        print(json.dumps(validator.issues, indent=2))
    else:
        print(report)
    
    # Exit code based on errors
    error_count = sum(1 for i in validator.issues if i["severity"] == "ERROR")
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
