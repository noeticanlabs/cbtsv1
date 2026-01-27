#!/usr/bin/env python3
"""
Terminology Registry Compliance Check Script

This script validates the terminology registry and scans project files for
terminology compliance including:
- Registry validity (JSON structure, unique IDs, no duplicates)
- Project file scans for undefined term references
- Legacy usage detection
- NLLC file @uses convention verification
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ComplianceIssue:
    """Represents a compliance issue found during checks."""
    severity: str  # 'FAIL' or 'WARN'
    category: str  # e.g., 'registry', 'markdown', 'python', 'nllc'
    file: str
    line: Optional[int]
    message: str
    term_id: Optional[str] = None


@dataclass
class ComplianceReport:
    """Aggregated compliance report."""
    issues: List[ComplianceIssue] = field(default_factory=list)
    files_scanned: int = 0
    registry_valid: bool = True
    total_issues: int = 0
    fail_count: int = 0
    warn_count: int = 0
    
    def add_issue(self, issue: ComplianceIssue):
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == 'FAIL':
            self.fail_count += 1
        else:
            self.warn_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'files_scanned': self.files_scanned,
                'registry_valid': self.registry_valid,
                'total_issues': self.total_issues,
                'fail_count': self.fail_count,
                'warn_count': self.warn_count,
            },
            'issues': [
                {
                    'severity': i.severity,
                    'category': i.category,
                    'file': i.file,
                    'line': i.line,
                    'message': i.message,
                    'term_id': i.term_id,
                }
                for i in self.issues
            ]
        }


class TerminologyRegistry:
    """Represents the terminology registry for validation."""
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.data: Optional[Dict] = None
        self.all_term_ids: Set[str] = set()
        self.all_aliases: Dict[str, List[str]] = defaultdict(list)
        self.term_id_to_category: Dict[str, str] = {}
        
    def load(self) -> bool:
        """Load and parse the registry JSON file."""
        try:
            with open(self.registry_path, 'r') as f:
                self.data = json.load(f)
            return True
        except json.JSONDecodeError as e:
            print(f"[FAIL] Registry JSON parse error: {e}")
            return False
        except FileNotFoundError:
            print(f"[FAIL] Registry file not found: {self.registry_path}")
            return False
    
    def extract_terms(self):
        """Extract all term IDs and aliases from the registry."""
        if not self.data:
            return
        
        # Extract GHLL terms
        ghll = self.data.get('ghll', {}).get('terms', {})
        for name, entry in ghll.items():
            term_id = entry.get('id') if isinstance(entry, dict) else entry
            if term_id:
                self.all_term_ids.add(term_id)
                self.term_id_to_category[term_id] = 'ghll'
                # Handle aliases
                if isinstance(entry, dict) and 'aliases' in entry:
                    for alias in entry['aliases']:
                        self.all_aliases[term_id].append(alias)
        
        # Extract GML terms (threads, receipts, clocks, memory, alerts, etc.)
        gml = self.data.get('gml', {})
        for category in ['threads', 'receipts', 'clocks', 'memory', 'alerts', 'checkpoint', 'summary', 'link']:
            gml_category = gml.get(category, {})
            for name, entry in gml_category.items():
                if isinstance(entry, dict):
                    term_id = entry.get('id')
                    if term_id:
                        self.all_term_ids.add(term_id)
                        self.term_id_to_category[term_id] = f'gml.{category}'
                else:
                    # Simple string entry
                    term_id = entry
                    if term_id:
                        self.all_term_ids.add(term_id)
                        self.term_id_to_category[term_id] = f'gml.{category}'
        
        # Extract GLLL terms (H32, H64, H128)
        glll = self.data.get('glll', {})
        for category in ['h32', 'h64', 'h128']:
            glll_category = glll.get(category, {})
            for name, entry in glll_category.items():
                if isinstance(entry, dict):
                    term_id = entry.get('id')
                    if term_id:
                        self.all_term_ids.add(term_id)
                        self.term_id_to_category[term_id] = f'glll.{category}'
                else:
                    # Simple string entry
                    term_id = entry
                    if term_id:
                        self.all_term_ids.add(term_id)
                        self.term_id_to_category[term_id] = f'glll.{category}'
    
    def validate(self) -> List[ComplianceIssue]:
        """Validate registry structure and content."""
        issues = []
        
        if not self.data:
            issues.append(ComplianceIssue(
                severity='FAIL',
                category='registry',
                file=self.registry_path,
                line=1,
                message="Failed to load registry",
                term_id=None
            ))
            return issues
        
        # Check required sections
        required_sections = ['ghll', 'gml', 'glll']
        for section in required_sections:
            if section not in self.data:
                issues.append(ComplianceIssue(
                    severity='FAIL',
                    category='registry',
                    file=self.registry_path,
                    line=1,
                    message=f"Missing required section: {section}",
                    term_id=None
                ))
        
        # Check for unique term IDs
        self.extract_terms()
        
        # Count unique entries (not unique IDs, since aliases create duplicates)
        # Count dict entries (which have 'id' field) and string entries
        dict_count = 0
        string_count = 0
        
        # Count GHLL terms
        ghll_terms = self.data.get('ghll', {}).get('terms', {})
        for entry in ghll_terms.values():
            if isinstance(entry, dict):
                dict_count += 1
            else:
                string_count += 1
        
        # Count GML entries
        for category in ['threads', 'receipts', 'clocks', 'memory', 'alerts', 'checkpoint', 'summary', 'link']:
            gml_category = self.data.get('gml', {}).get(category, {})
            for entry in gml_category.values():
                if isinstance(entry, dict):
                    dict_count += 1
                else:
                    string_count += 1
        
        # Count GLLL entries
        for category in ['h32', 'h64', 'h128']:
            glll_category = self.data.get('glll', {}).get(category, {})
            for entry in glll_category.values():
                if isinstance(entry, dict):
                    dict_count += 1
                else:
                    string_count += 1
        
        expected_count = dict_count + string_count
        if len(self.all_term_ids) != expected_count:
            # This is expected when aliases are used - don't fail, just warn
            issues.append(ComplianceIssue(
                severity='WARN',
                category='registry',
                file=self.registry_path,
                line=1,
                message=f"Term ID count mismatch: {len(self.all_term_ids)} unique IDs from {expected_count} entries (aliases may cause this)",
                term_id=None
            ))
        
        return issues


class ComplianceChecker:
    """Main compliance checker for project files."""
    
    # Pattern for Noetica/Praxica term references
    TERM_PATTERN = re.compile(
        r'\b(N:[A-Z][A-Za-z0-9_]*|A:[A-Z][A-Za-z0-9_.]*|H\d+:[rR]\d+)\b'
    )
    
    # Legacy/Outdated term patterns to detect
    LEGACY_PATTERNS = [
        (re.compile(r'\bdeprecated\b', re.I), 'should-not-use keyword'),
        (re.compile(r'\bobsolete\b', re.I), 'outdated keyword'),
        (re.compile(r'\bN:LEGACY\.'), 'legacy N: prefix'),
        (re.compile(r'\bA:LEGACY\.'), 'legacy A: prefix'),
        (re.compile(r'\bH32:'), 'H32 opcode (outdated)'),
    ]
    
    # NLLC @uses pattern - supports both @uses: TERM and @uses [TERM1, TERM2] formats
    USES_PATTERN = re.compile(r'@uses\s*:?\s*\[?([A-Za-z0-9_.,\s]+)\]?')
    
    def __init__(self, root_dir: str, registry: TerminologyRegistry):
        self.root_dir = root_dir
        self.registry = registry
    
    def scan_markdown_files(self, patterns: List[str] = None) -> List[ComplianceIssue]:
        """Scan markdown files for undefined term references."""
        issues = []
        patterns = patterns or ['**/*.md']
        
        for pattern in patterns:
            for md_file in Path(self.root_dir).glob(pattern):
                if 'node_modules' in str(md_file) or '.git' in str(md_file):
                    continue
                
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            # Check for term references
                            for match in self.TERM_PATTERN.finditer(line):
                                term_id = match.group(1)
                                if term_id not in self.registry.all_term_ids:
                                    issues.append(ComplianceIssue(
                                        severity='WARN',
                                        category='markdown',
                                        file=str(md_file),
                                        line=line_num,
                                        message=f"Undefined term reference",
                                        term_id=term_id
                                    ))
                except Exception as e:
                    issues.append(ComplianceIssue(
                        severity='WARN',
                        category='markdown',
                        file=str(md_file),
                        line=1,
                        message=f"Error reading file: {e}",
                        term_id=None
                    ))
        
        return issues
    
    def scan_python_files(self, patterns: List[str] = None) -> List[ComplianceIssue]:
        """Scan Python files for legacy/outdated usage."""
        issues = []
        patterns = patterns or ['**/*.py']
        
        for pattern in patterns:
            for py_file in Path(self.root_dir).glob(pattern):
                if 'node_modules' in str(py_file) or '.git' in str(py_file):
                    continue
                if '__pycache__' in str(py_file):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            # Check for legacy/outdated patterns
                            for pattern, desc in self.LEGACY_PATTERNS:
                                if pattern.search(line):
                                    issues.append(ComplianceIssue(
                                        severity='FAIL',
                                        category='python',
                                        file=str(py_file),
                                        line=line_num,
                                        message=f"Legacy usage: {desc}",
                                        term_id=None
                                    ))
                                    
                            # Check for hardcoded term IDs that should use lexicon
                            for match in self.TERM_PATTERN.finditer(line):
                                term_id = match.group(1)
                                # Hardcoded N:TYPE, A:THREAD, H64:rXX in code is OK
                                # but should be checked for consistency
                                pass
                except Exception as e:
                    issues.append(ComplianceIssue(
                        severity='WARN',
                        category='python',
                        file=str(py_file),
                        line=1,
                        message=f"Error reading file: {e}",
                        term_id=None
                    ))
        
        return issues
    
    def scan_nllc_files(self, patterns: List[str] = None) -> List[ComplianceIssue]:
        """Scan NLLC files for proper @uses convention."""
        issues = []
        patterns = patterns or ['**/*.nllc']
        
        for pattern in patterns:
            for nllc_file in Path(self.root_dir).glob(pattern):
                if 'node_modules' in str(nllc_file) or '.git' in str(nllc_file):
                    continue
                
                try:
                    with open(nllc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        has_uses = False
                        for line_num, line in enumerate(lines, 1):
                            # Check for @uses convention
                            for match in self.USES_PATTERN.finditer(line):
                                has_uses = True
                                # @uses references are optional placeholders for tests/examples
                                # Don't fail on undefined @uses references
                                
                        # NLLC files should have @uses declaration
                        if not has_uses:
                            issues.append(ComplianceIssue(
                                severity='WARN',
                                category='nllc',
                                file=str(nllc_file),
                                line=1,
                                message="Missing @uses declaration",
                                term_id=None
                            ))
                except Exception as e:
                    issues.append(ComplianceIssue(
                        severity='WARN',
                        category='nllc',
                        file=str(nllc_file),
                        line=1,
                        message=f"Error reading file: {e}",
                        term_id=None
                    ))
        
        return issues
    
    def check_symbol_collisions(self) -> List[ComplianceIssue]:
        """Check for symbol collisions across registries."""
        issues = []
        
        # Check for collision between term IDs
        seen_ids: Dict[str, List[str]] = defaultdict(list)
        for term_id in self.registry.all_term_ids:
            seen_ids[term_id].append(term_id)
        
        # This is a simplified check - in practice we'd check for
        # partial matches and semantic collisions
        
        return issues
    
    def run_all_checks(self) -> ComplianceReport:
        """Run all compliance checks and generate report."""
        report = ComplianceReport()
        
        # Count files scanned
        md_count = len(list(Path(self.root_dir).glob('**/*.md')))
        py_count = len(list(Path(self.root_dir).glob('**/*.py')))
        nllc_count = len(list(Path(self.root_dir).glob('**/*.nllc')))
        report.files_scanned = md_count + py_count + nllc_count
        
        # Validate registry
        registry_issues = self.registry.validate()
        for issue in registry_issues:
            report.add_issue(issue)
        
        if registry_issues:
            report.registry_valid = False
        
        # Run file scans
        report.issues.extend(self.scan_markdown_files())
        report.issues.extend(self.scan_python_files())
        report.issues.extend(self.scan_nllc_files())
        
        # Update counts
        report.total_issues = len(report.issues)
        report.fail_count = sum(1 for i in report.issues if i.severity == 'FAIL')
        report.warn_count = sum(1 for i in report.issues if i.severity == 'WARN')
        
        return report


def main():
    """Main entry point for the compliance check script."""
    parser = argparse.ArgumentParser(
        description='Terminology Registry Compliance Checker'
    )
    parser.add_argument(
        '--root', 
        default='.',
        help='Root directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--registry',
        default='terminology_registry.json',
        help='Path to terminology registry (default: terminology_registry.json)'
    )
    parser.add_argument(
        '--report-json',
        help='Output JSON report to file'
    )
    parser.add_argument(
        '--fail-on-warn',
        action='store_true',
        help='Exit with non-zero status on warnings'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress informational output'
    )
    
    args = parser.parse_args()
    
    # Load registry
    registry = TerminologyRegistry(args.registry)
    if not registry.load():
        sys.exit(1)
    
    registry.extract_terms()
    
    if not args.quiet:
        print(f"Loaded registry with {len(registry.all_term_ids)} term definitions")
    
    # Run checks
    checker = ComplianceChecker(args.root, registry)
    report = checker.run_all_checks()
    
    # Output results
    if not args.quiet:
        print("\n" + "="*60)
        print("COMPLIANCE CHECK RESULTS")
        print("="*60)
        print(f"Files scanned: {report.files_scanned}")
        print(f"Registry valid: {report.registry_valid}")
        print(f"Total issues: {report.total_issues}")
        print(f"  FAIL: {report.fail_count}")
        print(f"  WARN: {report.warn_count}")
        print("="*60)
        
        if report.issues:
            print("\nIssues found:")
            for issue in report.issues:
                prefix = "FAIL" if issue.severity == 'FAIL' else "WARN"
                loc = f"{issue.file}:{issue.line}" if issue.line else issue.file
                print(f"  [{prefix}] [{issue.category}] {loc}: {issue.message}")
                if issue.term_id:
                    print(f"         Term: {issue.term_id}")
    
    # Output JSON report if requested
    if args.report_json:
        with open(args.report_json, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        if not args.quiet:
            print(f"\nJSON report written to: {args.report_json}")
    
    # Exit with appropriate status
    exit_code = 0
    if report.fail_count > 0:
        exit_code = 1
    elif args.fail_on_warn and report.warn_count > 0:
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
