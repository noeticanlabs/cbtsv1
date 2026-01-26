#!/usr/bin/env python3
import json
import os

def main():
    receipts_file = "aeonic_receipts.jsonl"
    if not os.path.exists(receipts_file):
        print(f"No receipts file found at {receipts_file}")
        return

    print(f"Inspecting {receipts_file}...")
    
    total = 0
    accepted = 0
    rejected = 0
    reasons = {}
    
    with open(receipts_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
                
            event = data.get('event')
            if event == 'STEP_ACCEPT':
                accepted += 1
                total += 1
            elif event == 'STEP_REJECT':
                rejected += 1
                total += 1
                reason = data.get('rejection_reason', 'unknown')
                reasons[reason] = reasons.get(reason, 0) + 1
                
    print("-" * 40)
    print(f"Total Steps Attempted: {total}")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    
    if total > 0:
        rate = (accepted / total) * 100.0
        print(f"Acceptance Rate: {rate:.2f}%")
    
    if rejected > 0:
        print("-" * 40)
        print("Rejection Reasons:")
        for r, count in reasons.items():
            print(f"  {r}: {count}")
    print("-" * 40)

    if accepted > 0 and (rejected / total) < 0.1:
        print("SUCCESS: Excessive rejection issue appears resolved (rejection rate < 10%).")
    elif total == 0:
        print("WARNING: No steps found.")
    else:
        print("FAILURE: Rejection rate is still high.")

if __name__ == "__main__":
    main()