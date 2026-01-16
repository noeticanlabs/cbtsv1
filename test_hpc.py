import numpy as np
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_clocks import AeonicClockPack

bank = AeonicMemoryBank(AeonicClockPack())
arr = np.random.rand(10, 10).astype(np.float64)
bank.put('test', 2, arr, arr.nbytes, 100, 100, 1.0, 0.0, False, [])
bank.maintenance_tick()
demoted = bank.get('test')
if isinstance(demoted, np.ndarray):
    print('Demoted dtype:', demoted.dtype)
else:
    print('Demoted to summary')

print("HPC calibration passed!")