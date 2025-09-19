import numpy as np

def hartree_fock_spin_orbital(h1, V, n_electrons, D=None, max_iter=50, conv_tol=1e-8, mix=0.5):
    """
    General Hartree–Fock in spin-orbital basis (no spin symmetry assumed).
    
    Args:
        h1 : (N, N) ndarray
            One-electron integrals in spin-orbital basis.
        V : (N, N, N, N) ndarray
            Two-electron integrals in spin-orbital basis: (pq|rs).
        n_electrons : int
            Number of electrons (can be odd).
        D : (N, N) ndarray
            Initial guess of density matrix.
        max_iter : int
            Maximum SCF iterations.
        conv_tol : float
            Convergence tolerance for energy.
        mix : float
            Damping parameter for update.
    
    Returns:
        E_hf : float
            Hartree–Fock energy.
        C : (N, N) ndarray
            Molecular orbital coefficients.
        eps : (N,) ndarray
            Orbital energies.
        D : (N, N) ndarray
            Final density matrix.
    """
    N = h1.shape[0]
    n_occ = n_electrons

    def _1rdm(C):
        C_occ = C[:, :n_occ]
        return np.dot(C_occ,C_occ.T.conj())

    C = None
    eps = None
    if D is None:
        # Initial guess: diagonalize h1
        eps, C = np.linalg.eigh(h1)
        D = _1rdm(C) 

    E_old = 0.0
    for iteration in range(max_iter):
        # Build Fock matrix
        J = np.einsum("pqrs,sr->pq", V, D, optimize=True)
        K = np.einsum("prqs,sr->pq", V, D, optimize=True)
        F = h1 + J - K
        # Electronic energy
        E_elec = np.trace(np.dot((h1 + F),D))/2
        # Convergence check
        if abs(E_elec - E_old) < conv_tol:
            print(f"Converged in {iteration+1} iterations.")
            return E_elec, C, eps, D

        E_old = E_elec
        # Solve F C = C eps
        eps, Cnew = np.linalg.eigh(F)
        C = Cnew if C is None else (1-mix) * C + mix * Cnew
        D = _1rdm(C)
    raise RuntimeError("SCF did not converge")

# Example usage (toy data)
if __name__ == "__main__":
    N = 6
    n_electrons = 3
    h1 = np.random.rand(N, N)
    h1 = (h1 + h1.T) / 2  # make Hermitian
    V = np.random.rand(N, N, N, N)
    V = (V + V.transpose(1,0,3,2)) / 2  # symmetrize chemist's notation
    
    E, C, eps, D = hartree_fock_spin_orbital(h1, V, n_electrons)
    print("HF energy:", E)
