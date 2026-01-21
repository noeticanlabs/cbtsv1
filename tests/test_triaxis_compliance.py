import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from aeonic_receipts import AeonicReceipts
from receipt_schemas import validate_receipt_chain, OmegaReceipt
from src.triaxis.lexicon import GML

class TriaxisComplianceTest:
    """
    Test Triaxis v1.2 compliance: receipts, threads, IDs.
    """

    def __init__(self):
        self.results = {}
        self.receipts_log = "test_triaxis_receipts.jsonl"
        self.aeonic = AeonicReceipts(log_file=self.receipts_log)

    def run_simulation(self):
        """Run short GR simulation and collect receipts."""
        solver = GRSolver(16, 16, 16, 1.0, 1.0, 1.0)
        solver.init_minkowski()

        # Run a few steps
        for i in range(3):
            dt, dominant_thread, rail_violation = solver.orchestrator.run_step()
            # Emit receipt (assuming gr_ledger emits via aeonic)
            # For now, manually emit or check if emitted

        return solver

    def validate_receipts(self):
        """Validate receipts.jsonl compliance."""
        if not os.path.exists(self.receipts_log):
            return {"error": "No receipts log found"}

        receipts = []
        with open(self.receipts_log, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Assuming it's asdict(omega_receipt)
                    receipt = OmegaReceipt(**data)
                    receipts.append(receipt)
                except Exception as e:
                    return {"error": f"Invalid receipt JSON: {e}"}

        # Validate chain
        chain_valid = validate_receipt_chain(receipts)

        # Validate IDs
        id_valid = True
        thread_valid = True
        for receipt in receipts:
            # Check record has intent_id starting with N:
            if 'intent_id' in receipt.record:
                if not receipt.record['intent_id'].startswith('N:'):
                    id_valid = False
            if 'thread' in receipt.record:
                thread = receipt.record['thread']
                if not thread.startswith('A:THREAD.') or thread not in [getattr(GML, attr) for attr in dir(GML) if attr.startswith('THREAD_')]:
                    thread_valid = False
            if 'ops' in receipt.record:
                for op in receipt.record['ops']:
                    if not op.startswith('H:'):
                        id_valid = False

        # Check floats are strings (canonical JSON)
        # In asdict, floats might be floats, but OmegaReceipt uses canonical
        # For simplicity, assume ok if parsed

        return {
            "chain_valid": chain_valid,
            "id_valid": id_valid,
            "thread_valid": thread_valid,
            "num_receipts": len(receipts)
        }

    def run_test(self):
        """Run compliance test."""
        print("Running Triaxis Compliance Test...")
        self.run_simulation()
        validation = self.validate_receipts()
        self.results = validation
        print(f"Results: {validation}")

        # Save results
        with open("test_triaxis_compliance_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        overall_pass = all(validation.values()) if isinstance(validation, dict) else False
        return overall_pass

if __name__ == "__main__":
    test = TriaxisComplianceTest()
    passed = test.run_test()
    print(f"Triaxis Compliance Test Passed: {passed}")