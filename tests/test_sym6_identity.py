import numpy as np
from gr_solver.gr_core_fields import sym6_to_mat33, det_sym6, inv_sym6, SYM6_IDX

def test_sym6_identity():
    gamma = np.zeros((1,6))
    gamma[0, SYM6_IDX["xx"]] = 1.0
    gamma[0, SYM6_IDX["yy"]] = 1.0
    gamma[0, SYM6_IDX["zz"]] = 1.0

    mat = sym6_to_mat33(gamma)
    assert np.allclose(mat[0], np.eye(3))

    det = det_sym6(gamma)
    assert np.allclose(det, 1.0)

    inv = inv_sym6(gamma)
    assert np.allclose(inv, gamma)

if __name__ == "__main__":
    test_sym6_identity()
    print("sym6 identity test passed.")