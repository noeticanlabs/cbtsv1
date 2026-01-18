import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_clocks import AeonicClockPack
from aeonic_receipts import AeonicReceipts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySystemTest:
    """
    Test the accuracy and capability of the Aeonic Memory System.
    Tests put/get operations, TTL expiration, tier management, and maintenance.
    """

    def __init__(self):
        self.clocks = AeonicClockPack()
        self.receipts = AeonicReceipts()
        self.memory = AeonicMemoryBank(self.clocks, self.receipts)

    def test_basic_put_get(self):
        logger.info("Testing basic put/get operations...")

        # Put some data
        data = np.array([1, 2, 3, 4, 5])
        key = "test_data"
        bytes_est = data.nbytes
        ttl_l = 1000
        ttl_s = 100
        risk_score = 0.1
        tainted = False
        regime_hashes = []

        self.memory.put(key, 1, data, bytes_est, ttl_l, ttl_s, 10.0, risk_score, tainted, regime_hashes)

        # Get the data
        retrieved = self.memory.get(key)
        if retrieved is None:
            logger.error("Failed to retrieve data immediately after put")
            return False

        if not np.array_equal(retrieved, data):
            logger.error("Retrieved data does not match original")
            return False

        logger.info("Basic put/get test passed")
        return True

    def test_ttl_expiration(self):
        logger.info("Testing TTL expiration...")

        data = np.array([10, 20, 30])
        key = "ttl_test"
        self.memory.put(key, 1, data, data.nbytes, ttl_l=1, ttl_s=1, recompute_cost_est=5.0, risk_score=0.0, tainted=False, regime_hashes=[])

        # Advance clocks to expire
        for _ in range(5):
            self.clocks.tick_all()

        # Run maintenance to expire
        self.memory.maintenance_tick()

        retrieved = self.memory.get(key)
        if retrieved is not None:
            logger.error("Data should have expired but was retrieved")
            return False

        logger.info("TTL expiration test passed")
        return True

    def test_tier_management(self):
        logger.info("Testing tier management...")

        # Put data in different tiers
        for tier in [1, 2, 3]:
            data = np.array([tier] * 10)
            key = f"tier_{tier}_data"
            self.memory.put(key, tier, data, data.nbytes, ttl_l=100, ttl_s=50, recompute_cost_est=1.0, risk_score=0.0, tainted=False, regime_hashes=[])

        # Check retrieval from each tier
        for tier in [1, 2, 3]:
            key = f"tier_{tier}_data"
            retrieved = self.memory.get(key)
            if retrieved is None or retrieved[0] != tier:
                logger.error(f"Failed to retrieve or validate data from tier {tier}")
                return False

        logger.info("Tier management test passed")
        return True

    def test_maintenance(self):
        logger.info("Testing maintenance tick... (skipped - tick_all not available)")

        # Put some data
        data = np.array([99])
        key = "maintenance_test"
        self.memory.put(key, 1, data, data.nbytes, ttl_l=100, ttl_s=50, recompute_cost_est=1.0, risk_score=0.0, tainted=False, regime_hashes=[])

        # Run maintenance
        self.memory.maintenance_tick()

        # Check retrieval still works
        retrieved = self.memory.get(key)
        if retrieved is None or retrieved[0] != 99:
            logger.error("Data lost after maintenance")
            return False

        logger.info("Maintenance test passed")
        return True

    def run(self):
        logger.info("Running Memory System Test...")

        tests = [
            self.test_basic_put_get,
            self.test_ttl_expiration,
            self.test_tier_management,
            self.test_maintenance
        ]

        passed = 0
        for test in tests:
            if test():
                passed += 1
            else:
                logger.error(f"Test {test.__name__} failed")

        success = passed == len(tests)
        if success:
            logger.info(f"All {passed}/{len(tests)} memory tests passed")
        else:
            logger.error(f"Only {passed}/{len(tests)} memory tests passed")

        return success

if __name__ == "__main__":
    test = MemorySystemTest()
    passed = test.run()
    print(f"Memory System Test Passed: {passed}")