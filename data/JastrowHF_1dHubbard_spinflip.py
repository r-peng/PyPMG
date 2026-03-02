"""
Generalized Hartree–Fock (GHF) + Slater–Jastrow variational optimization for the 1D Hubbard model
with an on-site spin-flip (transverse) term:
    H_sf = -h_x * sum_i ( c†_{i↓} c_{i↑} + c†_{i↑} c_{i↓} )

- Exact diagonalization (ED) comparison in the fixed-total-N sector
- Alternating optimization: (1) Jastrow parameters, (2) Spinor orbital rotations
"""

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brent


# =============================
# Utilities: bits & neighbors
# =============================
def bitcount(x: int) -> int:
    return int(x.bit_count())


def combos_bits(L, k):
    # all integers with exactly k bits set among L bits
    for occ in it.combinations(range(L), k):
        b = 0
        for i in occ:
            b |= 1 << i
        yield b


def neighbor_pairs(L, pbc=True):
    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 1:
        bonds.append((L - 1, 0))
    return bonds


def fermion_hop_sign_adjacent(bits, i, j, L, pbc=True):
    # We only use this for nearest neighbors (i,j)
    if abs(i - j) == 1:
        return +1.0
    if pbc and {i, j} == {0, L - 1}:
        # wrap bond: sign = (-1)^(# occupied on sites 1..L-2)
        if L <= 2:
            return +1.0
        mask_between = 0
        for k in range(1, L - 1):
            mask_between |= 1 << k
        parity = bitcount(bits & mask_between) & 1
        return -1.0 if parity else +1.0
    raise ValueError("fermion_hop_sign_adjacent called on non-nearest neighbors")


def apply_hop(bits, i, j):
    # remove from i, add to j
    return bits ^ (1 << i) ^ (1 << j)


# =============================
# Basis with fixed total particle number N (allows spin flips)
# =============================
def build_basis_totalN(L, N_total):
    basis = []
    for k_up in range(max(0, N_total - L), min(L, N_total) + 1):
        k_dn = N_total - k_up
        for bu in combos_bits(L, k_up):
            for bd in combos_bits(L, k_dn):
                basis.append((bu, bd))
    index = {st: i for i, st in enumerate(basis)}
    return basis, index


# =============================
# One-body: hopping
# =============================
def hopping_matrix(L, t=1.0, pbc=True):
    T = np.zeros((L, L), float)
    for i in range(L - 1):
        T[i, i + 1] = T[i + 1, i] = -t
    if pbc and L > 1:
        T[0, L - 1] = T[L - 1, 0] = -t
    return T


# =============================
# Spin-flip fermionic signs (block ordering: all ↑ then all ↓)
# =============================
def spinflip_sign_u2d(bu, bd, i, L):
    # annihilate u(i), then create d(i)
    # exponent = (num up before i) + (num up total - 1) + (num down before i)
    c_up_before = bitcount(bu & ((1 << i) - 1))
    c_dn_before = bitcount(bd & ((1 << i) - 1))
    expnt = c_up_before + (bitcount(bu) - 1) + c_dn_before
    return -1.0 if (expnt & 1) else +1.0


def spinflip_sign_d2u(bu, bd, i, L):
    # annihilate d(i), then create u(i)
    # exponent = (num up total) + (num down before i) + (num up before i)
    c_up_before = bitcount(bu & ((1 << i) - 1))
    c_dn_before = bitcount(bd & ((1 << i) - 1))
    expnt = bitcount(bu) + c_dn_before + c_up_before
    return -1.0 if (expnt & 1) else +1.0


# =======================================
# Many-body sparse structure (hops + spin flips)
# =======================================
def precompute_mb_struct_totalN(L, t, N_total, pbc=True, hflip=0.0):
    basis, index = build_basis_totalN(L, N_total)
    nb = len(basis)
    dcounts = np.zeros(nb, int)
    bonds = neighbor_pairs(L, pbc=pbc)

    from_idx, to_idx, weights = [], [], []

    for col, (bu, bd) in enumerate(basis):
        dcounts[col] = bitcount(bu & bd)

        # --- kinetic hops (↑)
        for i, j in bonds:
            if ((bu >> i) & 1) and not ((bu >> j) & 1):
                sgn = fermion_hop_sign_adjacent(bu, i, j, L, pbc)
                row = index[(apply_hop(bu, i, j), bd)]
                from_idx.append(col)
                to_idx.append(row)
                weights.append(-t * sgn)
            if ((bu >> j) & 1) and not ((bu >> i) & 1):
                sgn = fermion_hop_sign_adjacent(bu, j, i, L, pbc)
                row = index[(apply_hop(bu, j, i), bd)]
                from_idx.append(col)
                to_idx.append(row)
                weights.append(-t * sgn)

        # --- kinetic hops (↓)
        for i, j in bonds:
            if ((bd >> i) & 1) and not ((bd >> j) & 1):
                sgn = fermion_hop_sign_adjacent(bd, i, j, L, pbc)
                row = index[(bu, apply_hop(bd, i, j))]
                from_idx.append(col)
                to_idx.append(row)
                weights.append(-t * sgn)
            if ((bd >> j) & 1) and not ((bd >> i) & 1):
                sgn = fermion_hop_sign_adjacent(bd, j, i, L, pbc)
                row = index[(bu, apply_hop(bd, j, i))]
                from_idx.append(col)
                to_idx.append(row)
                weights.append(-t * sgn)

        # --- on-site spin flips
        if hflip != 0.0:
            for i in range(L):
                # up -> down
                if ((bu >> i) & 1) and not ((bd >> i) & 1):
                    bu2, bd2 = bu ^ (1 << i), bd ^ (1 << i)
                    sgn = spinflip_sign_u2d(bu, bd, i, L)
                    row = index[(bu2, bd2)]
                    from_idx.append(col)
                    to_idx.append(row)
                    weights.append(-hflip * sgn)
                # down -> up
                if ((bd >> i) & 1) and not ((bu >> i) & 1):
                    bu2, bd2 = bu ^ (1 << i), bd ^ (1 << i)
                    sgn = spinflip_sign_d2u(bu, bd, i, L)
                    row = index[(bu2, bd2)]
                    from_idx.append(col)
                    to_idx.append(row)
                    weights.append(-hflip * sgn)

    return (
        basis,
        np.array(dcounts),
        np.array(from_idx),
        np.array(to_idx),
        np.array(weights, float),
    )


# =============================
# Energies (variational expectation using precomputed structure)
# =============================
def energy_expectation_fast(psi, dcounts, from_idx, to_idx, weights, U):
    den = float(np.vdot(psi, psi).real) + 1e-30
    num_diag = float(U) * float(np.sum(dcounts * (np.abs(psi) ** 2)))
    num_off = float(np.sum(weights * np.conj(psi[to_idx]) * psi[from_idx]).real)
    return (num_diag + num_off) / den


# =============================
# Dense ED Hamiltonian (with spin flips)
# =============================
def build_hubbard_dense_with_spinflip(basis, L, t=1.0, U=4.0, pbc=True, hflip=0.0):
    nb = len(basis)
    H = np.zeros((nb, nb), float)
    bonds = neighbor_pairs(L, pbc=pbc)
    index = {st: i for i, st in enumerate(basis)}

    for col, (bu, bd) in enumerate(basis):
        # onsite U
        H[col, col] += U * bitcount(bu & bd)

        # up hops
        for i, j in bonds:
            if ((bu >> i) & 1) and not ((bu >> j) & 1):
                sgn = fermion_hop_sign_adjacent(bu, i, j, L, pbc)
                row = index[(apply_hop(bu, i, j), bd)]
                H[row, col] += -t * sgn
            if ((bu >> j) & 1) and not ((bu >> i) & 1):
                sgn = fermion_hop_sign_adjacent(bu, j, i, L, pbc)
                row = index[(apply_hop(bu, j, i), bd)]
                H[row, col] += -t * sgn

        # down hops
        for i, j in bonds:
            if ((bd >> i) & 1) and not ((bd >> j) & 1):
                sgn = fermion_hop_sign_adjacent(bd, i, j, L, pbc)
                row = index[(bu, apply_hop(bd, i, j))]
                H[row, col] += -t * sgn
            if ((bd >> j) & 1) and not ((bd >> i) & 1):
                sgn = fermion_hop_sign_adjacent(bd, j, i, L, pbc)
                row = index[(bu, apply_hop(bd, j, i))]
                H[row, col] += -t * sgn

        # on-site spin flips
        if hflip != 0.0:
            for i in range(L):
                # up -> down
                if ((bu >> i) & 1) and not ((bd >> i) & 1):
                    row = index[(bu ^ (1 << i), bd ^ (1 << i))]
                    sgn = spinflip_sign_u2d(bu, bd, i, L)
                    H[row, col] += -hflip * sgn
                # down -> up
                if ((bd >> i) & 1) and not ((bu >> i) & 1):
                    row = index[(bu ^ (1 << i), bd ^ (1 << i))]
                    sgn = spinflip_sign_d2u(bu, bd, i, L)
                    H[row, col] += -hflip * sgn

    H = 0.5 * (H + H.T)
    return H


# =============================
# Jastrow–Slater amplitudes (spinor version)
# =============================
def jastrow_logfactor(bu, bd, L, nbar, g=0.0, v_by_range=None, pbc=True):
    """
    log J(C) = -g Σ_i n_{i↑}n_{i↓}  - 1/2 Σ_{i,j} v_{|i-j|} (n_i - nbar)(n_j - nbar)
    v_by_range: dict like {1: v1, 2: v2, ...}. If None -> only on-site Gutzwiller.
    """
    dcount = bitcount(bu & bd)
    val = -g * dcount
    if v_by_range:
        n_i = np.array([((bu >> i) & 1) + ((bd >> i) & 1) for i in range(L)], float)
        x = n_i - nbar
        v = np.zeros((L, L), float)
        for r, vr in v_by_range.items():
            if r <= 0:
                continue
            if pbc:
                for i in range(L):
                    j = (i + r) % L
                    v[i, j] += vr
                    v[j, i] += vr
            else:
                for i in range(L - r):
                    j = i + r
                    v[i, j] += vr
                    v[j, i] += vr
        val += -0.5 * float(x @ v @ x)
    return val


def jastrow_slater_amplitudes_ghf(
    L, basis, V_spinor, N_e, nbar, g=0.0, v_by_range=None, pbc=True
):
    Phi = V_spinor[:, :N_e]  # 2L x N_e
    psi = np.zeros(len(basis), complex)
    for k, (bu, bd) in enumerate(basis):
        rows = []
        # up rows: 0..L-1
        for i in range(L):
            if (bu >> i) & 1:
                rows.append(i)
        # down rows: L..2L-1
        for i in range(L):
            if (bd >> i) & 1:
                rows.append(L + i)
        M = Phi[rows, :]  # N_e x N_e
        det_spinor = np.linalg.det(M)

        lj = jastrow_logfactor(bu, bd, L, nbar, g=g, v_by_range=v_by_range, pbc=pbc)
        psi[k] = np.exp(lj) * det_spinor
    return psi


# =============================
# Generalized Hartree–Fock (GHF) mean-field
# =============================
def ghf_orbitals(
    L, U, t=1.0, N_e=None, hflip=0.0, pbc=True, mix=0.5, tol=1e-6, maxiter=500, seed=0
):
    assert N_e is not None, "Provide total particle number N_e"
    rng = np.random.default_rng(seed)

    # initial densities ~ N_e/(2L) per spin with tiny stagger; small transverse coherence
    n_up = (N_e / (2.0 * L)) * np.ones(L)
    n_dn = (N_e / (2.0 * L)) * np.ones(L)
    stag = 0.03 * np.array([1 if i % 2 == 0 else -1 for i in range(L)], float)
    n_up += +stag
    n_dn += -stag
    n_up += (-n_up.sum() + N_e / 2.0) / L
    n_dn += (-n_dn.sum() + N_e / 2.0) / L

    kappa = 1e-3 * (rng.standard_normal(L) + 1j * rng.standard_normal(L))

    T = hopping_matrix(L, t=t, pbc=pbc)
    I = np.eye(L)

    for itn in range(maxiter):
        # Assemble 2L x 2L mean-field Hamiltonian
        Huu = T + np.diag(U * n_dn)
        Hdd = T + np.diag(U * n_up)
        Hud = (hflip * I) - np.diag(U * np.conj(kappa))
        Hdu = (hflip * I) - np.diag(U * kappa)
        H = np.block([[Huu, Hud], [Hdu, Hdd]])

        e, V = np.linalg.eigh(H)

        # Occupy the lowest N_e spinors
        Phi = V[:, :N_e]  # 2L x N_e
        Phi_u = Phi[:L, :]
        Phi_d = Phi[L:, :]

        n_up_new = np.sum(np.abs(Phi_u) ** 2, axis=1)
        n_dn_new = np.sum(np.abs(Phi_d) ** 2, axis=1)
        kappa_new = np.sum(np.conj(Phi_u) * Phi_d, axis=1)  # <c†_u c_d>

        # Mix
        n_up = (1 - mix) * n_up + mix * n_up_new
        n_dn = (1 - mix) * n_dn + mix * n_dn_new
        kappa = (1 - mix) * kappa + mix * kappa_new

        # Convergence
        err = max(
            np.max(np.abs(n_up - n_up_new)),
            np.max(np.abs(n_dn - n_dn_new)),
            np.max(np.abs(kappa - kappa_new)),
        )
        if err < tol:
            break

    # HF energy with proper DC correction (hflip is one-body)
    E_occ_sum = float(np.sum(e[:N_e]))
    E_dc = U * float(np.sum(n_up * n_dn - np.abs(kappa) ** 2))
    E_tot = E_occ_sum - E_dc

    return dict(V=V, e=e, n_up=n_up, n_dn=n_dn, kappa=kappa, E_tot=E_tot, iters=itn + 1)


# =============================
# Column Givens rotation (acts on orbitals/columns)
# =============================
def apply_givens_columns(V, idx_occ, idx_virt, theta):
    # rotate columns: (occ, virt) <- G(theta) * (occ, virt)
    c, s = np.cos(theta), np.sin(theta)
    Va = V[:, idx_occ].copy()
    Vb = V[:, idx_virt].copy()
    V[:, idx_occ] = c * Va + s * Vb
    V[:, idx_virt] = -s * Va + c * Vb
    # If V was orthonormal, it remains orthonormal.


# =============================
# Alternating optimization pieces (GHF spinors)
# =============================
def optimize_jastrow_for_orbitals_ghf(
    L, basis, V_spinor, N_e, nbar, pbc, dcounts, from_idx, to_idx, weights, U, x0=None
):
    def obj(p):
        g, v1, v2, v3 = map(float, p)
        psi = jastrow_slater_amplitudes_ghf(
            L,
            basis,
            V_spinor,
            N_e,
            nbar,
            g=g,
            v_by_range={1: v1, 2: v2, 3: v3},
            pbc=pbc,
        )
        return energy_expectation_fast(psi, dcounts, from_idx, to_idx, weights, U)

    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0, 0.0], float)

    res = minimize(
        obj, x0, method="Powell", options=dict(maxiter=200, xtol=1e-3, ftol=1e-6)
    )
    g_opt, v1_opt, v2_opt, v3_opt = map(float, res.x)
    return float(res.fun), g_opt, v1_opt, v2_opt, v3_opt


def optimize_orbitals_given_jastrow_ghf(
    L,
    basis,
    N_e,
    nbar,
    pbc,
    dcounts,
    from_idx,
    to_idx,
    weights,
    U,
    V_init,
    g,
    v_by_range,
    theta_max=0.3,
    max_sweeps=10,
    tol=1e-9,
):
    currV = V_init.copy()

    def E_from_V(Vfull):
        psi = jastrow_slater_amplitudes_ghf(
            L, basis, Vfull, N_e, nbar, g=g, v_by_range=v_by_range, pbc=pbc
        )
        return energy_expectation_fast(psi, dcounts, from_idx, to_idx, weights, U)

    E_curr = E_from_V(currV)

    for _ in range(max_sweeps):
        improved = False
        Ldim = currV.shape[0]
        best_delta, bestV = 0.0, None

        def E_of_theta(theta, h, p_abs):
            Vtrial = currV.copy()
            apply_givens_columns(Vtrial, h, p_abs, theta)
            return E_from_V(Vtrial)

        for h in range(N_e):  # occupied columns
            for p_abs in range(N_e, Ldim):  # virtual columns
                f = lambda th: E_of_theta(th, h, p_abs)
                try:
                    th_opt = brent(f, brack=(-theta_max, 0.0, +theta_max), tol=1e-3)
                except Exception:
                    cand = [
                        (-theta_max, f(-theta_max)),
                        (0.0, f(0.0)),
                        (theta_max, f(theta_max)),
                    ]
                    th_opt = min(cand, key=lambda z: z[1])[0]

                E_try = f(th_opt)
                delta = E_curr - E_try
                if delta > best_delta + 1e-12:
                    Vtrial = currV.copy()
                    apply_givens_columns(Vtrial, h, p_abs, th_opt)
                    best_delta, bestV = delta, Vtrial

        if bestV is not None and best_delta > tol:
            currV, E_curr, improved = bestV, E_curr - best_delta, True
        if not improved:
            break

    return currV, E_curr


# =============================
# Main: U scan with Alt-Opt + ED
# =============================
def main():
    # --- Model / system ---
    L = 6
    t = 1.0
    PBC = True

    # Choose filling (works for half-filling or away from it)
    N_up = 2
    N_dn = 2
    N_e = N_up + N_dn
    nbar = N_e / L

    # Transverse spin-flip strength h_x (set 0.0 to recover standard Hubbard)
    HFLIP = 0.5

    # --- Precompute many-body structure (fixed total N) ---
    basis, dcounts, from_idx, to_idx, weights = precompute_mb_struct_totalN(
        L, t, N_e, pbc=PBC, hflip=HFLIP
    )

    # U grid
    U_values = np.arange(0.5, 8.0 + 1e-12, 0.5)

    # Storage
    E_ED_list, E_HF_list, E_JS_list = [], [], []
    g_list, v1_list, v2_list, v3_list = [], [], [], []

    outer_cycles = 25

    for U in U_values:
        # (1) Fully converged GHF for baseline comparison
        hf_full = ghf_orbitals(
            L=L,
            U=float(U),
            t=t,
            N_e=N_e,
            hflip=HFLIP,
            pbc=PBC,
            mix=0.5,
            tol=1e-12,
            seed=1,
            maxiter=500,  # Converge fully
        )
        E_GHF_full = hf_full["E_tot"]

        # (2) Lightly converged GHF as a seed for alternating optimization
        hf_seed = ghf_orbitals(
            L=L,
            U=float(U),
            t=t,
            N_e=N_e,
            hflip=HFLIP,
            pbc=PBC,
            mix=0.5,
            tol=1e-12,
            seed=1,
            maxiter=5,  # Do not fully converge
        )
        V_spin = hf_seed["V"].copy()
        E_best = hf_seed["E_tot"]
        g_best, v1_best, v2_best, v3_best = 0.0, 0.0, 0.0, 0.0

        for cyc in range(outer_cycles):
            # (a) optimize Jastrow on current spinor orbitals
            EJ, g_opt, v1_opt, v2_opt, v3_opt = optimize_jastrow_for_orbitals_ghf(
                L, basis, V_spin, N_e, nbar, PBC, dcounts, from_idx, to_idx, weights, U
            )
            v_by = {1: v1_opt, 2: v2_opt, 3: v3_opt}

            # (b) optimize orbitals for fixed Jastrow (spinor rotations)
            V_new, E_new = optimize_orbitals_given_jastrow_ghf(
                L,
                basis,
                N_e,
                nbar,
                PBC,
                dcounts,
                from_idx,
                to_idx,
                weights,
                U,
                V_spin,
                g_opt,
                v_by,
                theta_max=0.5,
                max_sweeps=2,
                tol=1e-9,
            )

            if E_new + 1e-4 < E_best or cyc == 0:
                E_best = E_new
                g_best, v1_best, v2_best, v3_best = g_opt, v1_opt, v2_opt, v3_opt
                V_spin = V_new
            else:
                break

        # (3) Exact diagonalization for comparison (with spin flips)
        H_dense = build_hubbard_dense_with_spinflip(
            basis, L, t=t, U=float(U), pbc=PBC, hflip=HFLIP
        )
        E_ED = float(np.linalg.eigvalsh(H_dense)[0])

        # Record
        E_ED_list.append(E_ED)
        E_HF_list.append(E_GHF_full)  # Store the fully converged GHF
        E_JS_list.append(E_best)
        g_list.append(g_best)
        v1_list.append(v1_best)
        v2_list.append(v2_best)
        v3_list.append(v3_best)

        print(
            f"U={U:>4.1f} | ED={E_ED: .8f} | GHF={E_GHF_full: .8f} | Alt-JS={E_best: .8f} "
            f"@ (g={g_best:.3f}, v1={v1_best:.3f}, v2={v2_best:.3f}, v3={v3_best:.3f}) (cycles={cyc+1})"
        )

    # --- Save data to CSV ---
    data_to_save = np.array(
        [
            U_values,
            E_ED_list,
            E_JS_list,
            E_HF_list,
            g_list,
            v1_list,
            v2_list,
            v3_list,
        ]
    ).T
    header = "U,E_ED,E_Jastrow_GHF,E_GHF,g,v1,v2,v3"
    filename = f"JastrowHF_hubbard1d_tf_L{L}_n{nbar:.3f}.csv"
    np.savetxt(
        filename,
        data_to_save,
        delimiter=",",
        header=header,
        comments="",
    )

    # --- Plot ---
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(U_values, E_ED_list, "o-", label="ED")
    plt.plot(U_values, E_JS_list, "s-", label="Jastrow–GHF")
    plt.plot(U_values, E_HF_list, "^-", label="GHF")
    plt.xlabel("U / t")
    plt.ylabel("Ground-state energy")
    plt.title(f"1D Hubbard + spin-flip (L={L}, n={nbar:.3f}, PBC, h_x={HFLIP})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Inset: errors vs ED
    dGHF = np.array(E_HF_list) - np.array(E_ED_list)
    dJS = np.array(E_JS_list) - np.array(E_ED_list)
    ax = plt.gca()
    axins = ax.inset_axes([0.55, 0.12, 0.4, 0.4])
    axins.axhline(0.0, lw=0.8, alpha=0.5)
    axins.plot(U_values, dJS, "s-", label="JS - ED")
    axins.plot(U_values, dGHF, "^-", label="GHF - ED")
    axins.set_xlabel("U / t", fontsize=8)
    axins.set_ylabel("ΔE", fontsize=8)
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3)
    axins.legend(fontsize=7, frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig("E_vs_U_with_spinflip.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
