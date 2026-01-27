"""
GCAT1 Suite
"""

import logging
from src.core.gr_solver import GRSolver
from tests.test_0_bs2_evolution import Test0Bs2
from tests.test_1_mms_lite import Test1MmsLite
from tests.test_1a_discrete_defect import Test1A
from tests.test_1b_operator_convergence import Test1B
from tests.test_2_rcs import Test2Rcs
from tests.test_3_wave_dynamics import Test3
from tests.test_4_damping_calibration import Test4
from tests.test_5_gauge_invariance import Test5
from tests.test_6_shat_stability import Test6Shat
from tests.test_7_tss_stability import Test7Tss
from tests.test_8_jsc_consistency import Test8Jsc
from tests.test_9_bianchi_integrity import Test9Bianchi

class GCAT1Suite:
    def __init__(self, gr_solver: GRSolver, **kwargs):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)
        self.logger.info("GCAT1Suite initialized with GRSolver instance")
        # Handle optional parameters if needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run_all_tests(self):
        scorecard = {}
        scorecard['test_0_bs2'] = Test0Bs2(self.gr_solver).run()
        scorecard['test_1'] = Test1MmsLite(self.gr_solver).run()
        scorecard['test_1a'] = Test1A(self.gr_solver).run()
        scorecard['test_1b'] = Test1B(self.gr_solver).run()
        scorecard['test_2'] = Test2Rcs(self.gr_solver).run()
        scorecard['test_3'] = Test3(self.gr_solver).run()
        scorecard['test_4'] = Test4(self.gr_solver).run()
        scorecard['test_5'] = Test5(self.gr_solver).run()
        scorecard['test_6'] = Test6Shat(self.gr_solver).run()
        scorecard['test_7_tss'] = Test7Tss(self.gr_solver).run()
        scorecard['test_8_jsc'] = Test8Jsc(self.gr_solver).run()  # Optional
        scorecard['test_9_bianchi'] = Test9Bianchi(self.gr_solver).run()
        return scorecard

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)
    # Instantiate GRSolver with small grid N=8
    solver = GRSolver(8, 8, 8, dx=1.0, dy=1.0, dz=1.0)
    # Run only Test 1 (MMS-lite)
    result = Test1MmsLite(solver).run()
    # Convert numpy types to Python types for JSON
    result['passed'] = bool(result['passed'])
    print(result)
    # Save to receipts_gcat1_test1.json
    with open('receipts_gcat1_test1.json', 'w') as f:
        json.dump(result, f, indent=2)