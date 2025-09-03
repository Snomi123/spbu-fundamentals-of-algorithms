from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import DTypeLike

from practicum_7.lu import LinearSystemSolver
from src.common import NDArrayFloat


class LuSolverWithPermute(LinearSystemSolver):
    def __init__(self, A: NDArrayFloat, dtype: DTypeLike, permute: bool) -> None:
        super().__init__(A, dtype)
        self.L, self.U, self.P = self._decompose(permute)

    def solve(self, b: NDArrayFloat) -> NDArrayFloat:
        Pb = self.P @ b
        n = self.L.shape[0]
        y = np.zeros(n, dtype=self.dtype)
        
        for i in range(n):
            y[i] = Pb[i] - self.L[i, :i] @ y[:i]
        
        x = np.zeros_like(y)
        for i in reversed(range(n)):
            x[i] = (y[i] - self.U[i, i+1:] @ x[i+1:]) / self.U[i, i]
        
        return x

    def _decompose(self, permute: bool) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        U = self.A.astype(self.dtype).copy()
        n = U.shape[0]
        L = np.zeros((n, n), self.dtype)
        P = np.eye(n, dtype=self.dtype)
        
        for k in range(n):
            if permute:
                max_row = k + np.argmax(np.abs(U[k:, k]))
                if max_row != k:
                    U[[k, max_row], :] = U[[max_row, k], :]
                    P[[k, max_row]] = P[[max_row, k]]
                    L[[k, max_row], :k] = L[[max_row, k], :k]
            
            if U[k, k] == 0:
                raise np.linalg.LinAlgError("Zero pivot encountered")
            
            L[k+1:, k] = U[k+1:, k] / U[k, k]
            U[k+1:] -= L[k+1:, k, None] * U[k]
        
        np.fill_diagonal(L, 1.0)
        return L, U, P


def get_A_b(a_11: float, b_1: float) -> tuple[NDArrayFloat, NDArrayFloat]:
    A = np.array([[a_11, 1.0, -3.0], [6.0, 2.0, 5.0], [1.0, 4.0, -3.0]])
    b = np.array([b_1, 12.0, -39.0])
    return A, b


if __name__ == "__main__":
    p = 16  
    a_11 = 3 + 10 ** (-p)
    b_1 = -16 + 10 ** (-p)
    A, b = get_A_b(a_11, b_1)

    solver = LuSolverWithPermute(A, np.float64, permute=True)
    x = solver.solve(b)
    assert np.all(np.isclose(x, [1, -7, 4])), f"The anwser {x} is not accurate enough"
