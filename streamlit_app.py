"""Streamlit web interface for the uploaded axial pile FDA assignment.

Run locally: streamlit run streamlit_app.py
Host free: upload this directory to GitHub and deploy through share.streamlit.io.
"""
from __future__ import annotations

from io import BytesIO, StringIO
import csv
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pile_fda import PileInputs, solve_pile, plot_result

st.set_page_config(page_title="PileFDA | Axial Load Transfer", page_icon="🏗️", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1450px;}
    h1 {letter-spacing: -0.035em;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    .note {font-size: 0.90rem; color: #64748b;}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "2026 MATLAB handout": dict(N=50, L=10.0, Q=-50.0, d=0.5, t=0.05, E_GPa=200.0, kf=100000.0, kbase=1e12, base="spring"),
    "2025 Excel handout": dict(N=50, L=5.0, Q=0.05, d=0.5, t=0.05, E_GPa=200.0, kf=10000.0, kbase=100000.0, base="spring"),
    "Fixed-base demonstration": dict(N=50, L=10.0, Q=-50.0, d=0.5, t=0.05, E_GPa=200.0, kf=100000.0, kbase=1e12, base="fixed"),
}

st.title("PileFDA")
st.markdown("**Axial pile load-transfer analysis** · Finite-difference approximation vs. analytical solution")
st.caption("Based on Prof. Ahmed Elgamal's 2026 axial-pile finite-difference handout and the 2025 exact-solution/Excel notes. Educational elastic model, not a construction-design certification.")

with st.sidebar:
    st.header("Model inputs")
    preset_name = st.selectbox("Reference scenario", list(PRESETS), help="Switching reference scenarios initializes separate editable input fields.")
    a = PRESETS[preset_name]
    suffix = preset_name.replace(" ", "_")

    with st.form(key=f"analysis_{suffix}"):
        st.subheader("Pile and loading")
        L = st.number_input("Pile length L (m)", min_value=0.01, value=a["L"], step=0.5, format="%.3f")
        d = st.number_input("Outside diameter d (m)", min_value=0.001, value=a["d"], step=0.05, format="%.3f")
        t = st.number_input("Wall thickness t (m)", min_value=0.0001, value=a["t"], step=0.005, format="%.4f")
        E_GPa = st.number_input("Young's modulus E (GPa)", min_value=0.001, value=a["E_GPa"], step=10.0, format="%.3f")
        Q = st.number_input("Head force Q (kN)", value=a["Q"], step=10.0, format="%.5g",
                            help="Signed as in the 2026 MATLAB code: negative = compression; positive = tension. 2025 handout specifies +50 N = +0.05 kN.")
        N = st.number_input("Finite-difference segments N", min_value=2, max_value=1000, value=a["N"], step=5)

        st.subheader("Soil resistance")
        soil_label = st.selectbox("Shaft stiffness with depth", ["Constant", "Linear variation", "Two layers"])
        soil_mode = {"Constant":"uniform", "Linear variation":"linear", "Two layers":"two_layer"}[soil_label]
        kf = st.number_input("Shaft subgrade modulus at top (kPa/m)", min_value=0.0,
                             value=a["kf"], step=10000.0, format="%.4g")
        kf_toe = None
        interface = None
        if soil_mode != "uniform":
            kf_toe = st.number_input("Shaft modulus at toe / second layer (kPa/m)", min_value=0.0,
                                     value=a["kf"] * 2, step=10000.0, format="%.4g")
        if soil_mode == "two_layer":
            interface = st.number_input("Layer interface depth (m)", min_value=0.001,
                                        max_value=float(L - 0.001) if L > 0.002 else 0.001,
                                        value=L/2, step=0.5, format="%.3f")

        base_label = st.radio("Pile toe boundary", ["End-bearing spring", "Fixed base (u(L) = 0)"],
                              index=0 if a["base"] == "spring" else 1)
        base = "spring" if base_label.startswith("End") else "fixed"
        kbase = a["kbase"]
        if base == "spring":
            kbase = st.number_input("End-bearing subgrade modulus (kPa/m)",
                                    min_value=0.0, value=a["kbase"], format="%.5g")
        st.form_submit_button("Calculate pile response", type="primary", use_container_width=True)

try:
    params = PileInputs(N=int(N), L=L, Q=Q, d=d, t=t,
                        E=E_GPa * 1e6, kf=kf, kbase=kbase,
                        base=base, soil=soil_mode, kf_toe=kf_toe, interface_depth=interface)
    result = solve_pile(params)
except (ValueError, ArithmeticError, OverflowError) as exc:
    st.error(f"Please correct the inputs: {exc}")
    st.stop()

r = result
z = r["z_m"]

m1,m2,m3,m4 = st.columns(4)
m1.metric("Head displacement", f"{r['u_m'][0]*1000:.6g} mm")
m2.metric("Toe axial force", f"{r['tip_force_kN']:.6g} kN")
m3.metric("Displacement error", "Not applicable" if r["max_disp_err_mm"] is None else f"{r['max_disp_err_mm']:.3g} mm")
m4.metric("Force error", "Not applicable" if r["max_force_err_kN"] is None else f"{r['max_force_err_kN']:.3g} kN")

if r["u_exact_m"] is None:
    st.info("Depth-varying shaft stiffness is an extension of the handout. The FDA result is available, but the uniform-stiffness analytical formula is not valid for this profile; exact curves and error metrics are therefore omitted.")


def profile_plot(x, x_exact, title, xaxis):
    """White plot panels with explicitly dark labels, ticks, axes, and gridlines.

    Explicit font settings are important because the overall Streamlit page
    may be using a dark theme. Blue solid and red dashed/outlined markers
    distinguish the curves without relying on color alone.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=z, name="FDA numerical (solid)", mode="lines",
        line=dict(width=3.5, color="#15558A"),
        hovertemplate="%{x:.7g}<br>Depth: %{y:.3f} m<extra>FDA</extra>",
    ))
    if x_exact is not None:
        fig.add_trace(go.Scatter(
            x=x_exact, y=z, name="Exact analytical (dashed)", mode="lines",
            line=dict(width=2.8, color="#B42318", dash="dash"),
            hovertemplate="%{x:.7g}<br>Depth: %{y:.3f} m<extra>Exact</extra>",
        ))
        # A few open squares reveal the analytical curve when the two lines
        # almost completely overlap (as expected in a converged solution).
        step = max(1, len(z) // 13)
        fig.add_trace(go.Scatter(
            x=x_exact[::step], y=z[::step], mode="markers",
            marker=dict(symbol="square-open", size=8, color="#B42318",
                        line=dict(color="#B42318", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ))

    axis_style = dict(
        showline=True, linewidth=1.8, linecolor="#111827", mirror=True,
        ticks="outside", ticklen=6, tickwidth=1.6, tickcolor="#111827",
        tickfont=dict(size=14, color="#111827"),
        title_font=dict(size=16, color="#111827"),
        showgrid=True, gridcolor="#C5CCD6", gridwidth=1,
        zeroline=True, zerolinecolor="#64748B", zerolinewidth=1.3,
    )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=19, color="#111827"), x=0.03, y=0.98),
        xaxis=dict(**axis_style, title=xaxis, rangemode="tozero"),
        yaxis=dict(**axis_style, title="Depth (m)", autorange="reversed"),
        font=dict(family="Arial, sans-serif", size=14, color="#111827"),
        legend=dict(orientation="h", x=0.0, y=1.14, yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.97)",
                    font=dict(color="#111827", size=13)),
        margin=dict(l=65, r=28, t=95, b=70),
        height=550, paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_color="#111827", font_size=14),
    )
    if "k₀" in xaxis:
        fig.update_xaxes(tickformat=",.0f")
    return fig

c1,c2 = st.columns(2)
with c1:
    st.plotly_chart(profile_plot(r["u_m"] * 1000,
                                None if r["u_exact_m"] is None else r["u_exact_m"] * 1000,
                                "Displacement vs. depth", "Axial displacement (mm)"), use_container_width=True)
with c2:
    st.plotly_chart(profile_plot(r["F_kN"], r["F_exact_kN"],
                                "Axial force vs. depth", "Axial force (kN)"), use_container_width=True)

st.caption("Solid blue = finite differences; dashed red / open squares = analytical solution. Nearly overlapping curves indicate close agreement for the constant-stiffness example.")

with st.expander("View pile, soil, and transfer details", expanded=False):
    cc1,cc2,cc3,cc4 = st.columns(4)
    cc1.metric("Pipe cross-section", f"{params.area:.5f} m²")
    cc2.metric("EA", f"{params.EA:,.0f} kN")
    cc3.metric("Pile-tip area (plugged)", f"{params.toe_area:.5f} m²")
    cc4.metric("Toe spring stiffness", "Fixed" if base == "fixed" else f"{params.kend:,.5g} kN/m")
    st.caption("A vertical shaft-stiffness line is expected when the constant stiffness option is selected.")
    ss1,ss2 = st.columns(2)
    with ss1:
        st.plotly_chart(profile_plot(r["kf_kPa_per_m"], None, "Shaft stiffness profile", "k₀(z) (kPa/m)"), use_container_width=True)
    with ss2:
        st.plotly_chart(profile_plot(r["shaft_transfer_kN_per_m"], None, "Distributed soil resistance", "k(z)u(z) (kN/m)"), use_container_width=True)

records = pd.DataFrame({
    "Depth (m)": z,
    "FDA displacement (mm)": r["u_m"] * 1000,
    "FDA axial force (kN)": r["F_kN"],
    "Shaft modulus (kPa/m)": r["kf_kPa_per_m"],
    "Shaft transfer (kN/m)": r["shaft_transfer_kN_per_m"],
})
if r["u_exact_m"] is not None:
    records["Exact displacement (mm)"] = r["u_exact_m"] * 1000
    records["Exact axial force (kN)"] = r["F_exact_kN"]
    records["Displacement difference (mm)"] = (r["u_m"] - r["u_exact_m"]) * 1000
    records["Axial force difference (kN)"] = r["F_kN"] - r["F_exact_kN"]

st.subheader("Results and export")
st.dataframe(records, hide_index=True, use_container_width=True, height=255)
fig_out = plot_result(result)
plot_buf = BytesIO()
fig_out.savefig(plot_buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
plot_buf.seek(0)
import matplotlib.pyplot as plt
plt.close(fig_out)

d1,d2 = st.columns(2)
d1.download_button("Download results (CSV)", records.to_csv(index=False).encode("utf-8"),
                   file_name="pile_fda_results.csv", mime="text/csv", use_container_width=True)
d2.download_button("Download comparison plots (PNG)", plot_buf.getvalue(),
                   file_name="pile_fda_plots.png", mime="image/png", use_container_width=True)

with st.expander("Method, boundary conditions, and important assumptions", expanded=False):
    st.markdown(r"""
**Governing equation (linear Winkler model)**

$$EA\,u''(z)-k(z)u(z)=0,\qquad k(z)=k_0(z)\pi d$$

**Second-order finite difference at node** $i$:

$$u_{i-1}+\left[-2-\frac{k(z_i)(\Delta z)^2}{EA}\right]u_i+u_{i+1}=0$$

**Pile-head force** $F(0)=Q$; shaft force from the computed displacements:

$$F(z_i)=-EA\frac{u_{i+1}-u_{i-1}}{2\Delta z}$$

**Base**: $F(L)=K_{\mathrm{tip}}u(L)$ with $K_{\mathrm{tip}}=k_{\mathrm{base}}\pi d^2/4$,
 or alternatively $u(L)=0$ for the fixed-base case.

- The solution uses $N+3$ unknown displacements including ghost nodes, as in the 2026 MATLAB handout.
- The pile is assumed linear elastic, constant cross section, and axially loaded. The toe uses the **plugged** pile area.
- Shaft springs are **linear elastic**. Nonlinear API $t-z$ and $Q-z$ mobilization curves are **not yet implemented**.
- For constant shaft stiffness only, an exact analytical solution is displayed; the toe stiffness in its nondimensional coefficient uses **total toe stiffness**, not the soil modulus by itself.
- For a depth-varying soil profile, results are numerical only; do not use the constant-modulus exact curve as a reference.
- All calculations use **m**, **kN**, **kPa** and **kPa/m**. Compression is a negative signed load, following the 2026 MATLAB example.
    """)
    st.caption("Note: The 2025 example specifies Q = 50 N (+0.05 kN), whereas the separate 2026 MATLAB example uses Q = -50 kN. These are different loading scenarios and must not be mixed.")

st.caption("PileFDA | Classroom calculation aid • Check soil parameters, boundary assumptions, and units before interpreting results.")
