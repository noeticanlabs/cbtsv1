#!/usr/bin/env python3
"""
Comprehensive comparison script for GR solver tests: Python vs NLLC implementations.

This script compares test_comprehensive_gr_solver.py and test_comprehensive_gr_solver.nllc
across length, performance, and accuracy metrics.

Usage:
    python compare_gr_tests.py

Requirements:
- Python test: tests/test_comprehensive_gr_solver.py
- NLLC test: test_comprehensive_gr_solver.nllc + run_nllc_gr_test.py
- Dependencies: numpy, etc. (from requirements.txt)

Outputs:
- Comparison results in compare_gr_tests_results.json
- Console summary table
"""

import subprocess
import json
import time
import os
import sys
from typing import Dict, Any, Tuple, Optional
import re

class GRTestComparator:
    def __init__(self):
        self.python_cmd = [sys.executable, "tests/test_comprehensive_gr_solver.py", "--N", "8", "--dt", "0.1"]
        self.nllc_cmd = [sys.executable, "run_nllc_gr_test.py"]
        self.results = {}

    def run_command_with_timing(self, cmd: list, cwd: str = ".") -> Tuple[float, str, str, int]:
        """Run command and return (elapsed_time, stdout, stderr, returncode)"""
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
        except subprocess.TimeoutExpired:
            return float('inf'), "", "Timeout", -1
        except Exception as e:
            return float('inf'), "", str(e), -1

        elapsed = time.time() - start_time
        return elapsed, result.stdout, result.stderr, result.returncode

    def measure_code_length(self) -> Dict[str, Dict[str, int]]:
        """Measure code length using wc -l"""
        lengths = {}

        # Python file
        try:
            result = subprocess.run(["wc", "-l", "tests/test_comprehensive_gr_solver.py"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = int(result.stdout.strip().split()[0])
                lengths["python"] = {"total_lines": lines}
            else:
                lengths["python"] = {"total_lines": -1, "error": result.stderr}
        except Exception as e:
            lengths["python"] = {"total_lines": -1, "error": str(e)}

        # NLLC file
        try:
            result = subprocess.run(["wc", "-l", "test_comprehensive_gr_solver.nllc"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = int(result.stdout.strip().split()[0])
                lengths["nllc"] = {"total_lines": lines}
            else:
                lengths["nllc"] = {"total_lines": -1, "error": result.stderr}
        except Exception as e:
            lengths["nllc"] = {"total_lines": -1, "error": str(e)}

        return lengths

    def run_python_test(self) -> Dict[str, Any]:
        """Run Python test and parse results"""
        print("Running Python test...")
        elapsed, stdout, stderr, retcode = self.run_command_with_timing(self.python_cmd)

        result = {
            "execution_time": elapsed,
            "returncode": retcode,
            "stdout": stdout,
            "stderr": stderr
        }

        if retcode != 0:
            result["error"] = "Non-zero return code"
            return result

        # Parse JSON output
        try:
            with open("test_E1_N8.json", "r") as f:
                data = json.load(f)

            # Extract metrics
            summary = data.get("summary", {})
            multi_step = data.get("multi_step_evolution", {})

            result.update({
                "overall_passed": summary.get("overall_passed", False),
                "tests_passed": summary.get("tests_passed", 0),
                "eps_H_final": multi_step.get("final_eps_H"),
                "eps_M_final": multi_step.get("final_eps_M"),
                "eps_H_history": multi_step.get("eps_H_history", []),
                "eps_M_history": multi_step.get("eps_M_history", []),
                "execution_steps": len(multi_step.get("eps_H_history", []))
            })

        except Exception as e:
            result["parse_error"] = str(e)

        return result

    def run_nllc_test(self) -> Dict[str, Any]:
        """Run NLLC test and parse results"""
        print("Running NLLC test...")
        elapsed, stdout, stderr, retcode = self.run_command_with_timing(self.nllc_cmd)

        result = {
            "execution_time": elapsed,
            "returncode": retcode,
            "stdout": stdout,
            "stderr": stderr
        }

        if retcode != 0:
            result["error"] = "Non-zero return code"
            return result

        # Parse console output for metrics
        eps_H_values = []
        eps_M_values = []

        lines = stdout.split('\n')
        passed = True
        steps_completed = 0

        for line in lines:
            if "eps_H =" in line and "eps_M =" in line:
                # Extract eps_H and eps_M from lines like "Step 0 accepted: eps_H = 1.23 eps_M = 4.56"
                match = re.search(r'eps_H\s*=\s*([-\d.e]+).*eps_M\s*=\s*([-\d.e]+)', line)
                if match:
                    eps_H_values.append(float(match.group(1)))
                    eps_M_values.append(float(match.group(2)))
                    steps_completed += 1

            if "failed" in line.lower() or "error" in line.lower():
                passed = False

        # Also try to parse receipts if available
        receipts_data = {}
        try:
            with open("test_comprehensive_gr_solver_nllc_receipts.json", "r") as f:
                receipts = json.load(f)
                receipts_data = {"total_receipts": len(receipts)}
        except Exception:
            pass

        result.update({
            "overall_passed": passed,
            "eps_H_final": eps_H_values[-1] if eps_H_values else None,
            "eps_M_final": eps_M_values[-1] if eps_M_values else None,
            "eps_H_history": eps_H_values,
            "eps_M_history": eps_M_values,
            "execution_steps": steps_completed,
            "receipts": receipts_data
        })

        return result

    def calculate_accuracy_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics: residuals, pass/fail, stability"""
        metrics = {}

        eps_H_hist = data.get("eps_H_history", [])
        eps_M_hist = data.get("eps_M_history", [])

        if eps_H_hist:
            metrics["eps_H_final"] = eps_H_hist[-1]
            metrics["eps_H_mean"] = sum(eps_H_hist) / len(eps_H_hist)
            metrics["eps_H_std"] = (sum((x - metrics["eps_H_mean"])**2 for x in eps_H_hist) / len(eps_H_hist))**0.5
            metrics["eps_H_range"] = max(eps_H_hist) - min(eps_H_hist)
        else:
            metrics["eps_H_final"] = None
            metrics["eps_H_mean"] = None
            metrics["eps_H_std"] = None
            metrics["eps_H_range"] = None

        if eps_M_hist:
            metrics["eps_M_final"] = eps_M_hist[-1]
            metrics["eps_M_mean"] = sum(eps_M_hist) / len(eps_M_hist)
            metrics["eps_M_std"] = (sum((x - metrics["eps_M_mean"])**2 for x in eps_M_hist) / len(eps_M_hist))**0.5
            metrics["eps_M_range"] = max(eps_M_hist) - min(eps_M_hist)
        else:
            metrics["eps_M_final"] = None
            metrics["eps_M_mean"] = None
            metrics["eps_M_std"] = None
            metrics["eps_M_range"] = None

        metrics["passed"] = data.get("overall_passed", False)
        metrics["steps"] = len(eps_H_hist)

        return metrics

    def compare_metrics(self, python: Dict[str, Any], nllc: Dict[str, Any]) -> Dict[str, Any]:
        """Compare Python vs NLLC metrics with ratios and differences"""
        comparison = {}

        # Helper function for safe ratio/diff calculation
        def safe_ratio(a, b):
            if a is None or b is None or b == 0:
                return None
            return a / b

        def safe_diff(a, b):
            if a is None or b is None:
                return None
            return a - b

        # Length comparison
        py_lines = python.get("total_lines", 0)
        nllc_lines = nllc.get("total_lines", 0)
        comparison["length"] = {
            "python_lines": py_lines,
            "nllc_lines": nllc_lines,
            "ratio_python_to_nllc": safe_ratio(py_lines, nllc_lines)
        }

        # Performance comparison
        py_time = python.get("execution_time", float('inf'))
        nllc_time = nllc.get("execution_time", float('inf'))
        comparison["performance"] = {
            "python_time": py_time,
            "nllc_time": nllc_time,
            "ratio_python_to_nllc": safe_ratio(py_time, nllc_time),
            "time_difference": safe_diff(py_time, nllc_time)
        }

        # Accuracy comparison
        py_acc = self.calculate_accuracy_metrics(python)
        nllc_acc = self.calculate_accuracy_metrics(nllc)

        comparison["accuracy"] = {
            "python": py_acc,
            "nllc": nllc_acc,
            "eps_H_final_diff": safe_diff(py_acc.get("eps_H_final"), nllc_acc.get("eps_H_final")),
            "eps_M_final_diff": safe_diff(py_acc.get("eps_M_final"), nllc_acc.get("eps_M_final")),
            "eps_H_std_ratio": safe_ratio(py_acc.get("eps_H_std"), nllc_acc.get("eps_H_std")),
            "eps_M_std_ratio": safe_ratio(py_acc.get("eps_M_std"), nllc_acc.get("eps_M_std")),
            "both_passed": py_acc.get("passed", False) and nllc_acc.get("passed", False)
        }

        return comparison

    def run_comparison(self) -> Dict[str, Any]:
        """Run full comparison"""
        print("Starting GR test comparison...")

        # Measure code lengths
        print("Measuring code lengths...")
        lengths = self.measure_code_length()

        # Run tests
        python_result = self.run_python_test()
        nllc_result = self.run_nllc_test()

        # Combine results
        self.results = {
            "length": lengths,
            "python": python_result,
            "nllc": nllc_result,
            "comparison": self.compare_metrics({**lengths.get("python", {}), **python_result},
                                            {**lengths.get("nllc", {}), **nllc_result})
        }

        return self.results

    def print_summary_table(self):
        """Print ASCII table summary"""
        comp = self.results.get("comparison", {})

        print("\n" + "="*80)
        print("GR SOLVER TEST COMPARISON SUMMARY")
        print("="*80)

        # Length
        len_data = comp.get("length", {})
        print(f"{'Metric':<20} {'Python':<15} {'NLLC':<15} {'Ratio (Py/NLLC)':<15}")
        print("-"*65)
        print(f"{'Code Lines':<20} {len_data.get('python_lines', 'N/A'):<15} {len_data.get('nllc_lines', 'N/A'):<15} {len_data.get('ratio_python_to_nllc', 'N/A'):.2f}")

        # Performance
        perf = comp.get("performance", {})
        print(f"\n{'Execution Time (s)':<20} {perf.get('python_time', 'N/A'):<15.3f} {perf.get('nllc_time', 'N/A'):<15.3f} {perf.get('ratio_python_to_nllc', 'N/A'):.2f}")

        # Accuracy
        acc = comp.get("accuracy", {})
        print(f"\n{'Final eps_H':<20} {acc.get('python', {}).get('eps_H_final', 'N/A'):<15.3e} {acc.get('nllc', {}).get('eps_H_final', 'N/A'):<15.3e} {acc.get('eps_H_final_diff', 'N/A'):<15.3e}")
        print(f"{'Final eps_M':<20} {acc.get('python', {}).get('eps_M_final', 'N/A'):<15.3e} {acc.get('nllc', {}).get('eps_M_final', 'N/A'):<15.3e} {acc.get('eps_M_final_diff', 'N/A'):<15.3e}")
        print(f"{'eps_H Stability':<20} {acc.get('python', {}).get('eps_H_std', 'N/A'):<15.3e} {acc.get('nllc', {}).get('eps_H_std', 'N/A'):<15.3e} {acc.get('eps_H_std_ratio', 'N/A'):.2f}")
        print(f"{'eps_M Stability':<20} {acc.get('python', {}).get('eps_M_std', 'N/A'):<15.3e} {acc.get('nllc', {}).get('eps_M_std', 'N/A'):<15.3e} {acc.get('eps_M_std_ratio', 'N/A'):.2f}")
        print(f"{'Overall Pass':<20} {str(acc.get('python', {}).get('passed', False)):<15} {str(acc.get('nllc', {}).get('passed', False)):<15} {str(acc.get('both_passed', False)):<15}")

        print("\nNotes:")
        print("- Ratio > 1 means Python is larger/slower/less stable")
        print("- Stability measured as standard deviation of residuals over steps")
        print("- Positive difference means Python > NLLC")

    def save_results(self, filename: str = "compare_gr_tests_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    comparator = GRTestComparator()
    try:
        results = comparator.run_comparison()
        comparator.print_summary_table()
        comparator.save_results()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()