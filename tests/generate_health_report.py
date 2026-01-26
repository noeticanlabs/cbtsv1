#!/usr/bin/env python3
"""
System Health Report Generator
Aggregates receipts from GCAT-0.5, GCAT-1, and GCAT-2 into a single status document.
"""

import json
import datetime
import os

def load_receipt(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f), True
    return {}, False

def get_status(receipt, loaded, key='passed'):
    if not loaded:
        return "MISSING"
    return "PASS" if receipt.get(key) else "FAIL"

def main():
    # Load all receipts
    gcat05, gcat05_loaded = load_receipt('receipts_gcat0_5_lie_detector.json')
    gcat1, gcat1_loaded = load_receipt('receipts_gcat1_test1.json')
    gcat2_s2, gcat2_s2_loaded = load_receipt('receipts_gcat2_s2_summary.json')
    gcat2_s3, gcat2_s3_loaded = load_receipt('receipts_gcat2_s3_summary.json')

    # Check if GCAT-2 overall is loaded (requires both scenarios to be present to judge)
    gcat2_loaded = gcat2_s2_loaded and gcat2_s3_loaded

    # Determine overall status
    checks = [
        gcat05.get('passed', False),
        gcat1.get('passed', False),
        gcat2_s2.get('passed', False),
        gcat2_s3.get('passed', False)
    ]
    
    # System is healthy only if ALL checks pass and we actually ran tests
    overall_passed = all(checks) and len(checks) > 0

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "HEALTHY" if overall_passed else "UNHEALTHY",
        "summary": {
            "GCAT-0.5 (Lie Detector)": get_status(gcat05, gcat05_loaded),
            "GCAT-1 (Convergence)": get_status(gcat1, gcat1_loaded),
            "GCAT-2 (Scenarios)": "PASS" if (gcat2_loaded and gcat2_s2.get('passed') and gcat2_s3.get('passed')) else ("MISSING" if not gcat2_loaded else "FAIL")
        },
        "details": {
            "GCAT-0.5": {
                "diagnosis": gcat05.get('diagnosis', 'N/A'),
                "results": gcat05.get('results', {})
            },
            "GCAT-1": {
                "diagnosis": gcat1.get('diagnosis', 'N/A'),
                "p_obs": gcat1.get('metrics', {}).get('p_obs', 0.0)
            },
            "GCAT-2": {
                "Scenario 2": gcat2_s2.get('diagnosis', 'N/A'),
                "Scenario 3": gcat2_s3.get('diagnosis', 'N/A')
            }
        }
    }

    # Write report
    with open('system_health_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report generated. System Status: {report['status']}")

if __name__ == "__main__":
    main()