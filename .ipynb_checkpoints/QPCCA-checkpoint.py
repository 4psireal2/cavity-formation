from scipy.optimize import fmin
from typing import Dict, List, Tuple, Union, Callable, Optional, TYPE_CHECKING
import numpy as np

def _objective(alpha: np.ndarray, X: np.ndarray) -> float:
    """
    Compute objective function value.
    Parameters
    ----------
    alpha
        Vector of shape `((m - 1) ^ 2,)` containing the flattened and
        cropped rotation matrix ``rot_matrix[1:, 1:]``.
    X
        %(Q_sort)s
    Returns
    -------
    Current value of the objective function :math:`f = m - trace(S)`
    (Eq. 16 from [Roeblitz13]_).
    """
    # Dimensions.
    n, m = X.shape
    k = m - 1

    # Initialize rotation matrix.
    rot_mat = np.zeros((m, m), dtype=np.float64)

    # Sanity checks.
    if alpha.shape[0] != k ** 2:
        raise ValueError(
            "The shape of alpha doesn't match with the shape of X: "
            f"It is not a ({k}^2,)-vector, but of dimension {alpha.shape}. X is of shape `{X.shape}`."
        )

    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))

    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_mat[1:, 1:] = rot_crop_matrix
    rot_mat = _fill_matrix(rot_mat, X)

    # Compute value of the objective function.
    # from Matlab: optval = m - trace( diag(1 ./ A(1,:)) * (A' * A) )
    return m - np.trace(np.diag(1.0 / rot_mat[0, :]).dot(rot_mat.conj().T.dot(rot_mat)))  # type: ignore[no-any-return]

def _indexsearch(X: np.ndarray) -> np.ndarray:
    """
    Find a simplex structure in the data.
    Parameters
    ----------
    X
        %(Q_sort)s
    Returns
    -------
    Vector of shape `(m,)` with indices of data points that constitute the
    vertices of a simplex.
    """
    n, m = X.shape

    # Sanity check.
    if n < m:
        raise ValueError(
            f"The Schur vector matrix of shape {X.shape} has more columns than rows. "
            f"You can't get a {m}-dimensional simplex from {n} data vectors."
        )
    # Check if the first, and only the first eigenvector is constant.
    diffs = np.abs(np.max(X, axis=0) - np.min(X, axis=0))
    if not np.isclose(1.0 + diffs[0], 1.0, rtol=1e-6):
        raise ValueError(
            f"First Schur vector is not constant 1. This indicates that the Schur vectors "
            f"are incorrectly sorted. Cannot search for a simplex structure in the data. The largest deviation from 1 "
            f"is {diffs[0]}."
        )
    if not np.all(diffs[1:] > 1e-6):
        which = np.sum(diffs[1:] <= 1e-6)
        raise ValueError(
            f"{which} Schur vector(s) after the first one are constant. Probably the Schur vectors "
            "are incorrectly sorted. Cannot search for a simplex structure in the data."
        )

    # local copy of the eigenvectors
    ortho_sys = np.copy(X)

    index = np.zeros(m, dtype=np.int64)
    max_dist = 0.0

    # First vertex: row with largest norm.
    for i in range(n):
        dist = np.linalg.norm(ortho_sys[i, :])
        if dist > max_dist:
            max_dist = dist
            index[0] = i

    # Translate coordinates to make the first vertex the origin.
    ortho_sys -= np.ones((n, 1)).dot(ortho_sys[index[0], np.newaxis])
    # Would be shorter, but less readable: ortho_sys -= X[index[0], np.newaxis]

    # All further vertices as rows with maximum distance to existing subspace.
    for j in range(1, m):
        max_dist = 0.0
        temp = np.copy(ortho_sys[index[j - 1], :])
        for i in range(n):
            sclprod = ortho_sys[i, :].dot(temp)
            ortho_sys[i, :] -= sclprod * temp
            distt = np.linalg.norm(ortho_sys[i, :])
            if distt > max_dist:  # and i not in index[0:j]: #in _pcca_connected_isa() of pcca.py
                max_dist = distt
                index[j] = i
        ortho_sys /= max_dist

    return index

def _initialize_rot_matrix(X: np.ndarray) -> np.ndarray:
    """
    Initialize the rotation matrix.
    Parameters
    ----------
    X
        %(Q_sort)s
    Returns
    -------
    Initial (non-optimized) rotation matrix of shape `(m, m)`.
    """
    # Search start simplex vertices ('inner simplex algorithm').
    index = _indexsearch(X)

    # Local copy of the Schur vectors.
    # Xc = np.copy(X)

    # Raise or warn if condition number is (too) high.
    condition = np.linalg.cond(X[index, :])
    if condition >= (1.0 / EPS):
        raise ValueError(
            f"The condition number {condition} of the matrix of start simplex vertices "
            "X[index, :] is too high for safe inversion (to build the initial rotation matrix)."
        )
    if condition > 1e4:
        warnings.warn(
            f"The condition number {condition} of the matrix of start simplex vertices "
            "X[index, :] is quite high for safe inversion (to build the initial rotation matrix)."
        )

    # Compute transformation matrix rot_matrix as initial guess for local optimization (maybe not feasible!).
    return np.linalg.pinv(X[index, :])

def _gpcca_core(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Core of the G-PCCA [Reuter18]_ spectral clustering method with optimized memberships.
    Clusters the dominant `m` Schur vectors of a transition matrix.
    This algorithm generates a fuzzy clustering such that the resulting
    membership functions are as crisp (characteristic) as possible.
    Parameters
    ----------
    X
        %(Q_sort)s
    Returns
    -------
    Triple of the following:
    chi
        %(chi_ret)s
    rot_matrix
        %(rot_matrix_ret)s
    crispness
        %(crispness_ret)s
    """
    m = np.shape(X)[1]

    rot_matrix = _initialize_rot_matrix(X)

    rot_matrix, chi, fopt = _opt_soft(X, rot_matrix)

    # calculate crispness of the decomposition of the state space into m clusters
    crispness = (m - fopt) / m

    return chi, rot_matrix, crispness

def _fill_matrix(rot_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Make the rotation matrix feasible.
    Parameters
    ----------
    rot_matrix
        (Infeasible) rotation matrix of shape `(m, m)`.
    X
        %(Q_sort)s
    Returns
    -------
    Feasible rotation matrix of shape `(m, m)`.
    """
    n, m = X.shape

    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic.")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors.")

    # Compute first column of rot_mat by row sum condition.
    rot_matrix[1:, 0] = -np.sum(rot_matrix[1:, 1:], axis=1)

    # Compute first row of A by maximum condition.
    dummy = -np.dot(X[:, 1:], rot_matrix[1:, :])
    rot_matrix[0, :] = np.max(dummy, axis=0)

    # Reskale rot_mat to be in the feasible set.
    rot_matrix = rot_matrix / np.sum(rot_matrix[0, :])

    # Make sure, that there are no zero or negative elements in the first row of A.
    if np.any(rot_matrix[0, :] == 0):
        raise ValueError("First row of rotation matrix has elements = 0.")
    if np.min(rot_matrix[0, :]) < 0:
        raise ValueError("First row of rotation matrix has elements < 0.")

    return rot_matrix

def _opt_soft(X: np.ndarray, rot_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Optimize the G-PCCA rotation matrix such that the memberships are
    exclusively non-negative and compute the membership matrix.
    Parameters
    ----------
    X
        %(Q_sort)s
    rot_matrix
        Initial (non-optimized) rotation matrix of shape `(m, m)`.
    Returns
    -------
    Triple of the following:
    rot_matrix
        %(rot_matrix_ret)s
    chi
        %(chi_ret)s
    fopt
        Optimal value of the objective function :math:`f_{opt} = m - \\mathtt{trace}(S)`
        (Eq. 16 from [Roeblitz13]_).
    """  # noqa: D205, D400
    n, m = X.shape

    # Sanity checks.
    if not (rot_matrix.shape[0] == rot_matrix.shape[1]):
        raise ValueError("Rotation matrix isn't quadratic.")
    if not (rot_matrix.shape[0] == m):
        raise ValueError("The dimensions of the rotation matrix don't match with the number of Schur vectors.")
    if rot_matrix.shape[0] < 2:
        raise ValueError(f"Expected the rotation matrix to be at least of shape (2, 2), found {rot_matrix.shape}.")

    # Reduce optimization problem to size (m-1)^2 by cropping the first row and first column from rot_matrix
    rot_crop_matrix = rot_matrix[1:, 1:]

    # Now reshape rot_crop_matrix into a linear vector alpha.
    k = m - 1
    alpha = np.reshape(rot_crop_matrix, k ** 2)
    # TODO: Implement Gauss Newton Optimization to speed things up esp. for m > 10
    alpha, fopt, _, _, _ = fmin(_objective, alpha, args=(X,), full_output=True, disp=False)

    # Now reshape alpha into a (k,k)-matrix.
    rot_crop_matrix = np.reshape(alpha, (k, k))

    # Complete rot_mat to meet constraints (positivity, partition of unity).
    rot_matrix[1:, 1:] = rot_crop_matrix
    rot_matrix = _fill_matrix(rot_matrix, X)

    # Compute the membership matrix.
    chi = np.dot(X, rot_matrix)

    # Check for negative elements in chi and handle them.
    if np.any(chi < 0.0):
        if np.any(chi < -1e4 * EPS):
            min_el = np.min(chi)
            raise ValueError(f"Some elements of chi are significantly negative. The minimal element in chi is {min_el}")
        else:
            chi[chi < 0.0] = 0.0
            chi = np.true_divide(1.0, np.sum(chi, axis=1))[:, np.newaxis] * chi
            if not np.allclose(np.sum(chi, axis=1), 1.0, atol=1e-8, rtol=1e-5):
                dev = np.max(np.abs(np.sum(chi, axis=1) - 1.0))
                raise ValueError(
                    f"The rows of chi don't sum up to 1.0 after rescaling. Maximum deviation from 1 is {dev}"
                )

    return rot_matrix, chi, fopt