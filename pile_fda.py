"""Finite-difference axial pile solver based on Elgamal (2026) FDA handout.

Units: kN, m, kPa (=kN/m^2), kPa/m (=kN/m^3).
Sign convention: Q < 0 denotes pile compression (matches the 2026 MATLAB code).
Displacement and force carry the sign of the applied load.

The N+3-by-N+3 sparse system explicitly retains the two ghost nodes,
matching the handout. A fixed-base boundary condition is also supported.

To run as standalone code:
    python pile_fda.py
    python pile_fda.py --help
The companion `streamlit_app.py` imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class PileInputs:
    N: int = 50                 # Number of pile segments
    L: float = 10.0            # Pile length [m]
    Q: float = -50.0           # Applied pile-head axial force [kN]
    d: float = 0.5             # Outer diameter [m]
    t: float = 0.05            # Pipe wall thickness [m]
    E: float = 2.0e8          # Young's modulus [kPa], 200 GPa
    kf: float = 100000.0       # Soil subgrade modulus [kPa/m]
    kbase: float = 1.0e12     # Tip subgrade modulus [kPa/m]
    base: str = "spring"       # 'spring' or 'fixed'
    soil: str = "uniform"     # 'uniform', 'linear', 'two_layer'
    kf_toe: float | None = None
    interface_depth: float | None = None

    def validate(self) -> None:
        if not 2 <= self.N <= 2000 or int(self.N) != self.N:
            raise ValueError("N must be a whole number between 2 and 2000.")
        for key in ("L", "d", "t", "E"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
        if self.t >= self.d / 2:
            raise ValueError("Wall thickness must be smaller than half the outer diameter.")
        if not math.isfinite(self.Q):
            raise ValueError("Q must be finite.")
        for key in ("kf", "kbase"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
        if self.base not in ("spring", "fixed"):
            raise ValueError("Base must be 'spring' or 'fixed'.")
        if self.soil not in ("uniform", "linear", "two_layer"):
            raise ValueError("Soil must be 'uniform', 'linear', or 'two_layer'.")
        if self.soil != "uniform":
            if self.kf_toe is None or not math.isfinite(self.kf_toe) or self.kf_toe < 0:
                raise ValueError("Layer/toe modulus must be finite and nonnegative.")
        if self.soil == "two_layer":
            if self.interface_depth is None or not 0 < self.interface_depth < self.L:
                raise ValueError("Interface depth must be strictly between 0 and L.")
        if self.base == "spring" and self.kbase == 0 and self.kf == 0 and (
            self.soil == "uniform" or self.kf_toe == 0
        ):
            raise ValueError("Free tip and zero shaft stiffness leave a loaded pile unsupported.")

    @property
    def area(self) -> float:
        return math.pi * (self.d**2 - (self.d - 2 * self.t)**2) / 4.0

    @property
    def EA(self) -> float:
        return self.E * self.area  # kN

    @property
    def toe_area(self) -> float:
        # Plugged-pile base area, as specified in the MATLAB handout.
        return math.pi * self.d**2 / 4.0

    @property
    def kend(self) -> float:
        return self.kbase * self.toe_area  # kN/m


def subgrade_profile(p: PileInputs, z: np.ndarray) -> np.ndarray:
    """Return kf(z) in kPa/m at nodes (linear and two-layer are extensions)."""
    if p.soil == "uniform":
        return np.full_like(z, p.kf, dtype=float)
    if p.soil == "linear":
        return p.kf + (p.kf_toe - p.kf) * z / p.L
    return np.where(z < p.interface_depth, p.kf, p.kf_toe)


def exact_solution(p: PileInputs, z: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Analytical solution for constant k(z), including the two base alternatives.

    Correct end stiffness: K_tip = kbase * pi*d^2/4 [kN/m].
    This is required for dimensionally consistent Omega = K_tip/(EA*lambda).

    The shifted-depth form avoids cancellation of large hyperbolic terms.
    """
    if p.soil != "uniform":
        return None
    EA = p.EA
    lam = math.sqrt(p.kf * math.pi * p.d / EA)  # [1/m]
    if lam == 0:
        if p.base == "fixed":
            return p.Q * (p.L - z) / EA, np.full_like(z, p.Q)
        if p.kend == 0:
            raise ValueError("No shaft resistance or base resistance.")
        return p.Q * (p.L - z) / EA + p.Q / p.kend, np.full_like(z, p.Q)

    lamL = lam * p.L
    # For large lamL the /cosh(lamL) ratios are computed via scaled exponentials
    # rather than calling cosh(lamL), which can overflow.
    s = lam * (p.L - z)
    def cosh_ratio(v: np.ndarray) -> np.ndarray:
        return (np.exp(v - lamL) + np.exp(-v - lamL)) / (1 + np.exp(-2 * lamL))
    def sinh_ratio(v: np.ndarray) -> np.ndarray:
        return (np.exp(v - lamL) - np.exp(-v - lamL)) / (1 + np.exp(-2 * lamL))

    cr = cosh_ratio(s)
    sr = sinh_ratio(s)
    if p.base == "fixed":
        # u(L)=0, Q=-EA*du/dz at z=0
        return p.Q / (EA * lam) * sr, p.Q * cr

    omega = p.kend / (EA * lam)
    divisor = math.tanh(lamL) + omega
    return (p.Q / (EA * lam) * (cr + omega * sr) / divisor,
            p.Q * (sr + omega * cr) / divisor)


def solve_pile(p: PileInputs) -> dict:
    """Solve the handout's central finite differences using sparse linear algebra.

    Unknown vector: [u_-1, u_0, ..., u_N, u_(N+1)] with N+3 entries.
    First row: -u_-1 + u_1 = -2*dz*Q/EA.
    Interior physical nodes i=0..N:
        u_(i-1) + B_i*u_i + u_(i+1) = 0.
    Last row, spring: -u_(N-1) + C_N*u_N + u_(N+1) = 0.
    Last row, fixed: u_N = 0.
    """
    p.validate()
    z = np.linspace(0.0, p.L, p.N + 1)
    dz = p.L / p.N
    kf_z = subgrade_profile(p, z)
    k_z = kf_z * math.pi * p.d  # [kPa], modulus for shaft * pile circumference
    B = -2.0 - (dz**2 / p.EA) * k_z
    CN = 2.0 * dz * p.kend / p.EA
    size = p.N + 3

    K = lil_matrix((size, size), dtype=np.float64)
    f = np.zeros(size, dtype=np.float64)
    K[0, 0], K[0, 2] = -1.0, 1.0
    f[0] = -2.0 * dz * p.Q / p.EA
    for j in range(p.N + 1):
        r = j + 1  # matrix row representing physical node j
        K[r, j], K[r, j + 1], K[r, j + 2] = 1.0, B[j], 1.0

    if p.base == "spring":
        K[-1, -3], K[-1, -2], K[-1, -1] = -1.0, CN, 1.0
    else:
        # Replaces the toe force-spring condition with u_N=0.
        # The interior equation at node N still determines its ghost node.
        K[-1, -2] = 1.0

    K = K.tocsr()
    with np.errstate(all="raise"):
        u_all = spsolve(K, f)
    if not np.all(np.isfinite(u_all)):
        raise ArithmeticError("Solver returned non-finite values.")

    u = u_all[1:-1]  # u_0..u_N
    F = -p.EA * (u_all[2:] - u_all[:-2]) / (2.0 * dz)  # [kN] n=0..N
    spring_transfer = k_z * u  # [kN/m]: signed shaft transfer per unit depth
    analytic = exact_solution(p, z)
    if analytic is None:
        u_exact, F_exact = None, None
        disp_err = force_err = None
    else:
        u_exact, F_exact = analytic
        disp_err = float(np.max(np.abs(u - u_exact)) * 1000)  # mm
        force_err = float(np.max(np.abs(F - F_exact)))  # kN
    residual = float(np.linalg.norm(K @ u_all - f, np.inf))
    return {
        "inputs": p,
        "z_m": z,
        "dz_m": dz,
        "u_m": u,
        "F_kN": F,
        "kf_kPa_per_m": kf_z,
        "shaft_transfer_kN_per_m": spring_transfer,
        "u_exact_m": u_exact,
        "F_exact_kN": F_exact,
        "max_disp_err_mm": disp_err,
        "max_force_err_kN": force_err,
        "equation_residual": residual,
        "top_force_kN": float(F[0]),
        "tip_force_kN": float(F[-1]),
        "tip_reaction_kN": (float(p.kend * u[-1]) if p.base == "spring" else float(F[-1])),
    }


def export_csv(result: dict, path: str | Path) -> None:
    keys = ["Depth (m)", "FDA displacement (mm)", "FDA axial force (kN)",
            "Shaft subgrade modulus (kPa/m)", "Shaft transfer (kN/m)",
            "Exact displacement (mm)", "Exact axial force (kN)"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(keys)
        for i, z in enumerate(result["z_m"]):
            writer.writerow([
                z, 1000 * result["u_m"][i], result["F_kN"][i],
                result["kf_kPa_per_m"][i], result["shaft_transfer_kN_per_m"][i],
                "" if result["u_exact_m"] is None else 1000 * result["u_exact_m"][i],
                "" if result["F_exact_kN"] is None else result["F_exact_kN"][i],
            ])


def plot_result(result: dict):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    z = result["z_m"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    # Axes share y-limits: invert only once, or twice would cancel the reversal.
    ax[0].invert_yaxis()
    for a in ax:
        a.set_facecolor("white")
        a.grid(True, color="#dce3ed", linewidth=0.65)
        a.set_axisbelow(True)
        a.tick_params(labelsize=10)
    ax[0].plot(1000 * result["u_m"], z, linewidth=2.2, color="#0f4c81", label="FDA")
    ax[1].plot(result["F_kN"], z, linewidth=2.2, color="#0f4c81", label="FDA")
    if result["u_exact_m"] is not None:
        ax[0].plot(1000 * result["u_exact_m"], z, "--", linewidth=1.9,
                   color="#d58b23", label="Exact")
        ax[1].plot(result["F_exact_kN"], z, "--", linewidth=1.9,
                   color="#d58b23", label="Exact")
    ax[0].set_title("Axial displacement", weight="bold", fontsize=13)
    ax[1].set_title("Axial force", weight="bold", fontsize=13)
    ax[0].set_xlabel("Displacement (mm)")
    ax[1].set_xlabel("Axial force (kN)")
    ax[0].set_ylabel("Depth below head (m)")
    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("N", "L", "Q", "d", "t", "E", "kf", "kbase"):
        parser.add_argument("--" + field, type=int if field == "N" else float,
                            default=getattr(PileInputs(), field))
    parser.add_argument("--base", choices=["spring", "fixed"], default="spring")
    parser.add_argument("--soil", choices=["uniform", "linear", "two_layer"], default="uniform")
    parser.add_argument("--kf-toe", type=float, default=None)
    parser.add_argument("--interface-depth", type=float, default=None)
    parser.add_argument("--out", default="pile_fda_output", help="Output filename prefix")
    args = parser.parse_args()
    p = PileInputs(N=args.N, L=args.L, Q=args.Q, d=args.d, t=args.t,
                   E=args.E, kf=args.kf, kbase=args.kbase, base=args.base,
                   soil=args.soil, kf_toe=args.kf_toe,
                   interface_depth=args.interface_depth)
    r = solve_pile(p)
    out = Path(args.out)
    export_csv(r, str(out) + ".csv")
    fig = plot_result(r)
    fig.savefig(str(out) + ".png", dpi=210)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(json.dumps({"parameters": asdict(p),
                      "pile_head_displacement_mm": 1000 * r["u_m"][0],
                      "pile_tip_displacement_mm": 1000 * r["u_m"][-1],
                      "pile_head_force_kN": r["top_force_kN"],
                      "pile_tip_force_kN": r["tip_force_kN"],
                      "max_displacement_error_mm": r["max_disp_err_mm"],
                      "max_force_error_kN": r["max_force_err_kN"],
                      "max_equation_residual": r["equation_residual"]}, indent=2))
    print(f"Saved {out}.png and {out}.csv")


if __name__ == "__main__":
    main()
