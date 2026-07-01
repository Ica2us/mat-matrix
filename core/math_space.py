"""
core/math_space.py
==================
Compositional Data (CoDA) geometry transformations for material formulation data.

Concrete implementation of :class:`contracts.CompositionalMathProtocol`.
Class name MUST be ``CompositionalMathTransformer`` per project convention.

All random sampling reads ``contracts.GLOBAL_RANDOM_STATE`` to guarantee
deterministic reproducibility.
"""

from typing import cast
import numpy as np
import numpy.typing as npt
from contracts import CompositionalMathProtocol


def _helmert_contrast_matrix(d: int) -> np.ndarray:
    """
    Build the **standard (orthonormal) Helmert contrast matrix** of size ``d x d``.

    The Helmert matrix :math:`H_d` is constructed as:

    - Row 1: all entries = :math:`1/\\sqrt{d}`  (the sum-contrast row).
    - Row :math:`i` (for :math:`i = 2, \\dots, d`):
      - First :math:`i-1` entries = :math:`1 / \\sqrt{i \\, (i-1)}`
      - Entry :math:`i` = :math:`-(i-1) / \\sqrt{i \\, (i-1)}`
      - Remaining entries = 0.

    Removing the first row yields the :math:`(d-1) \\times d` contrast matrix
    used by the ILR transform.

    Parameters
    ----------
    d : int
        Number of parts (columns) in the composition.

    Returns
    -------
    np.ndarray
        Shape ``(d, d)`` orthonormal Helmert matrix.
    """
    H = np.zeros((d, d))
    # sum-contrast row
    H[0, :] = 1.0 / np.sqrt(d)

    for i in range(2, d + 1):
        row_idx = i - 1
        k = 1.0 / np.sqrt(i * (i - 1))
        H[row_idx, :i - 1] = k
        H[row_idx, i - 1] = -(i - 1) * k
        # entries beyond i remain 0
    return H


class CompositionalMathTransformer(CompositionalMathProtocol):
    """
    Deterministic geometric transformations between the simplex :math:`S^d`
    (compositional space) and the unconstrained real space :math:`\\mathbb{R}^{d-1}`.

    Uses multiplicative zero replacement (deterministic regularisation) followed
    by the Isometric Log-Ratio (ILR) transform via an orthonormal Helmert basis.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def multiplicative_zero_replacement(
        self, X: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """
        Replace every zero entry in *X* with ``eps`` and rescale each row's
        non-zero entries proportionally so that every row sums to exactly 1.

        Fully vectorised — no Python row-loop.  Handles three cases in a
        single pass:

        * **No zeros anywhere** — identity (return copy unchanged).
        * **All-zero rows** — replaced by uniform ``1 / n_cols``.
        * **Mixed rows** (some zeros, some non-zeros) — zeros -> ``eps``,
          original non-zeros scaled by ``(1 - n_zeros * eps) / orig_sum``.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, n_components)``.  Values should be non-negative.
        eps : float, optional
            Small positive replacement value (default ``1e-6``).

        Returns
        -------
        np.ndarray
            Transformed array, same shape, each row summing to 1.
        """
        if eps <= 0:
            raise ValueError("eps must be strictly positive")

        X = np.asarray(X, dtype=np.float64)
        n_rows, n_cols = X.shape

        # Guard: if eps is so large that even a single zero-replacement
        # could push the scaling factor negative (worst case = all columns
        # are zero), abort early with a clear message.
        if n_cols * eps >= 1.0:
            raise ValueError(
                f"eps ({eps}) is too large for {n_cols} components: "
                f"n_cols * eps = {n_cols * eps} >= 1.0. "
                "Choose a smaller eps such that n_cols * eps < 1.0."
            )

        # Detect zeros per row
        zero_mask = X <= 0
        n_zeros = np.sum(zero_mask, axis=1)  # shape (n,)

        # Fast path: no zeros anywhere
        if not np.any(n_zeros > 0):
            return X.copy()

        result = X.copy()

        # --- Case 2: all-zero rows -> uniform distribution ---
        all_zero = n_zeros == n_cols
        if np.any(all_zero):
            result[all_zero] = 1.0 / n_cols

        # --- Case 3: mixed rows (0 < n_zeros < n_cols) ---
        mixed = (n_zeros > 0) & (n_zeros < n_cols)
        if np.any(mixed):
            # Replace zero entries with eps (on mixed rows only)
            mixed_zero_mask = mixed[:, np.newaxis] & zero_mask
            result[mixed_zero_mask] = eps

            # Sum of original non-zero values per mixed row
            # read from *original* X, not result, so we aren't contaminated
            # by the eps we just wrote
            orig_nonzero = mixed[:, np.newaxis] & (~zero_mask)
            orig_nonzero_sum = np.sum(X * orig_nonzero, axis=1)  # (n,)

            # Scale factor: shrink original non-zeros so that
            #   scale * orig_sum + n_zeros * eps = 1
            # When 1 - n_zeros * eps <= 0 the non-zero part cannot absorb
            # the eps additions without becoming negative -> fall back to
            # uniform distribution for those rows.
            cap = (1.0 - n_zeros.astype(np.float64) * eps)
            cap_non_positive = cap <= 0

            if np.any(cap_non_positive):
                # These mixed rows become uniform
                result[cap_non_positive] = 1.0 / n_cols
                # Exclude them from further scaling
                mixed = mixed & (~cap_non_positive)
                if not np.any(mixed):
                    return result

            # Recalc for surviving mixed rows
            if np.any(mixed):
                orig_nonzero = mixed[:, np.newaxis] & (~zero_mask)
                orig_nonzero_sum = np.sum(X * orig_nonzero, axis=1)
                # Safety: orig_nonzero_sum is guaranteed > 0 for surviving
                # mixed rows (otherwise the row would be all-zero and already
                # handled).  The guard exists only to silence numpy's
                # divide-by-zero warning on rows not in `mixed`.
                with np.errstate(divide="ignore"):
                    scale = np.where(
                        orig_nonzero_sum > 0,
                        cap / orig_nonzero_sum,
                        0.0,
                    )
                result[orig_nonzero] *= np.broadcast_to(
                    scale[:, np.newaxis], result.shape
                )[orig_nonzero]

        return result

    def ilr_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Isometric Log-Ratio transform.

        Projects the ``(n, d)`` composition matrix *X* (rows on the simplex)
        to an ``(n, d-1)`` unconstrained real matrix via the orthonormal
        Helmert contrast basis.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_samples, n_components)``, strictly positive values
            (after zero replacement).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_components - 1)``.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if d < 2:
            raise ValueError("Need at least 2 components for ILR transform")

        H = _helmert_contrast_matrix(d)
        # Drop the first (sum-contrast) row -> shape (d-1, d)
        contrast = H[1:, :]

        # clr = log(x / g(x)) where g(x) is geometric mean
        logX = np.log(X)
        # clr_i = log(x_i) - mean(log(X), axis=1)
        clr = logX - np.mean(logX, axis=1, keepdims=True)

        # ILR = clr @ contrast.T
        Z = clr @ contrast.T
        return cast(npt.NDArray[np.float64], Z)

    def inverse_ilr_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Inverse Isometric Log-Ratio transform.

        Recovers the ``(n, d)`` simplex composition from the ``(n, d-1)``
        unconstrained real representation *Z*.

        Parameters
        ----------
        Z : np.ndarray
            Shape ``(n_samples, n_components - 1)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_components)``, each row summing to 1.
        """
        Z = np.asarray(Z, dtype=np.float64)
        n, k = Z.shape
        d = k + 1

        H = _helmert_contrast_matrix(d)
        contrast = H[1:, :]  # (d-1, d)

        # Recover CLR: clr = Z @ contrast
        clr = Z @ contrast

        # Recover composition: x_i = exp(clr_i) / sum(exp(clr))
        exp_clr = np.exp(clr)
        X = exp_clr / np.sum(exp_clr, axis=1, keepdims=True)
        return cast(npt.NDArray[np.float64], X)