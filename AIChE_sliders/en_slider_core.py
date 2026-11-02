# -*- coding: utf-8 -*-
'''
Shared steady 1D electroneutral Nernst--Planck (ENNP) model.

Put this file beside en_slider_dimensional.py and en_slider_Da_only.py.
The plots show only four ionic concentrations and the reaction rate.

Nondimensional model:
    J_i = -d_i (dC_i/dx - z_i C_i E)
    dJ_H/dx = dJ_OH/dx = -Da C_H C_OH
    dJ_Na/dx = dJ_Cl/dx = 0
    C_Cl = C_H + C_Na - C_OH
    J_Cl = J_H + J_Na - J_OH

State: [C_H, C_Na, C_OH, a_H, a_OH], where a_i = J_i/d_i.
Unknown constant BVP parameter: a_Na.

All concentration values are scaled by C_star = c0; all diffusivities
by D_star = max(D_H, D_Na, D_OH, D_Cl). When the dimensional sliders
change C_star or D_star, the corresponding reservoir concentrations
and species diffusivities scale together, preserving their ratios.
'''

#%% Imports
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp


#%% Physical / model parameters

# Domain length
L = 5e-4                        # [=] m

# Boundary concentrations
c0 = 1.0                        # [=] mol/m^3
cs = 0.2                        # [=] mol/m^3

# Boundary potentials (unused in the zero-current electroneutral model)
phi_0 = 0.0                     # [=] V
phi_L = 0.0                     # [=] V

# Diffusivities
D_H  = 9.31e-9                   # [=] m^2/s
D_Na = 1.33e-9                   # [=] m^2/s
D_Cl = 2.03e-9                   # [=] m^2/s
D_OH = 5.27e-9                   # [=] m^2/s

# Valencies
z_H = 1
z_Na = 1
z_Cl = -1
z_OH = -1

# Vacuum permittivity (unused without the Poisson equation)
eps_0 = 8.8541878128e-12        # [=] F/m

# Relative permittivity of water near room temperature
eps_r = 78.3                     # [=] dimensionless

# Electrolyte permittivity (unused without the Poisson equation)
permittivity = eps_r * eps_0     # [=] F/m

# Second-order volumetric rate constant: R = k c_H c_OH
# NOTE: k must have units m^3/(mol s), NOT mol/s.
k = 1.3e11                      # [=] m^3/(mol s)

# Faraday constant (not needed for this dimensionless zero-current closure)
Faraday = 96485.33212           # [=] C/mol

# Ideal gas constant
R_gas = 8.314462618             # [=] J/(mol K)

# Absolute temperature
Temp = 298.15                   # [=] K


#%% Numerical parameters
EPS = 1e-16                     # Floor for division by a concentration
cs_seed = 0.2                   # Reserved for other continuation workflows
n_continuation_steps = 60       # Maximum number of Da continuation steps
make_plots = True               # Toggle for launching the plotting scripts

N_MESH = 130                    # Starting BVP mesh points
N_PLOT = 450                    # Base plot points; add refined BVP nodes
BVP_TOL = 2e-4
MAX_NODES = 15000


#%% Plot style
plot_style = {
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180
}

colors = {
    'H': '#FF3A3A',
    'Na': '#F77A13',
    'OH': '#3E07F4',
    'Cl': '#0DB51E',
    'R': '#B60C86'
}


#%% Fixed dimensionless reference ratios
C_star_ref = c0
D_star_ref = max(D_H, D_Na, D_OH, D_Cl)
SALT_RATIO = cs / C_star_ref
D_REL = np.array([D_H, D_Na, D_OH, D_Cl]) / D_star_ref
DA_REFERENCE = k * C_star_ref * L**2 / D_star_ref


#%% Governing equations and boundary conditions
def ennp_ode(x, y, p, Da):
    '''Vectorized first-order ODE system with shape (5, len(x)).'''
    cH, cNa, cOH, aH, aOH = y
    dH, dNa, dOH, dCl = D_REL
    aNa = p[0]
    aCl = (dH * aH + dNa * aNa - dOH * aOH) / dCl

    # Electroneutral, zero-current electric-field closure, E = dpsi/dx.
    E = (aH + aNa - aOH - aCl) / (2.0 * np.maximum(cH + cNa, EPS))
    reaction = Da * cH * cOH

    return np.vstack((E * cH - aH,
                      E * cNa - aNa,
                      -E * cOH - aOH,
                      -reaction / dH,
                      -reaction / dOH))


def ennp_bc(ya, yb, p):
    '''Six boundary residuals for five state variables and one unknown a_Na.'''
    return np.array([ya[0] - 1.0, ya[1] - SALT_RATIO, ya[2],
                     yb[0], yb[1] - 1.0, yb[2] - 1.0])


def initial_guess(x):
    '''Concentration profiles that already match the reservoir conditions.'''
    y = np.zeros((5, x.size))
    y[0] = 1.0 - x
    y[1] = SALT_RATIO + (1.0 - SALT_RATIO) * x
    y[2] = x
    y[3] = 1.0
    y[4] = -1.0
    p = np.array([SALT_RATIO - 1.0])
    return y, p


def _solve_one(Da, x, y, p):
    sol = solve_bvp(lambda xx, yy, pp: ennp_ode(xx, yy, pp, Da),
                    ennp_bc, x, y, p=p, tol=BVP_TOL, max_nodes=MAX_NODES)
    if not sol.success:
        raise RuntimeError(f'BVP failed at Da = {Da:.3g}: {sol.message}')
    return sol


def solve_ennp(Da):
    '''Solve via geometric continuation from Da=0 to the requested Da.'''
    if not np.isfinite(Da) or Da < 0.0:
        raise ValueError('Da must be finite and nonnegative.')

    x = np.linspace(0.0, 1.0, N_MESH)
    y, p = initial_guess(x)
    sol = _solve_one(0.0, x, y, p)

    if Da > 0.0:
        first = min(Da, 1e-3)
        if Da > first:
            n_steps = min(n_continuation_steps,
                          max(2, int(np.ceil(4.0 * np.log10(Da / first))) + 1))
            path = np.geomspace(first, Da, n_steps)
        else:
            path = np.array([Da])
        for value in path:
            sol = _solve_one(value, sol.x, sol.y, sol.p)

    return sol


#%% Reconstruct and cache profiles
@lru_cache(maxsize=80)
def _profile_for_key(log_da):
    '''Cache a solved profile keyed by log10(Da), or -inf for Da=0.'''
    Da = 0.0 if np.isneginf(log_da) else 10.0**float(log_da)
    sol = solve_ennp(Da)

    # Include all adaptively refined BVP nodes: a uniform grid alone can
    # entirely miss the narrow reaction peak at very large Da.
    x = np.unique(np.concatenate((np.linspace(0.0, 1.0, N_PLOT), sol.x)))
    cH, cNa, cOH = sol.sol(x)[:3]
    cCl = cH + cNa - cOH
    concentrations = np.vstack((cH, cNa, cOH, cCl))
    reaction = Da * cH * cOH

    if not np.all(np.isfinite(concentrations)) or not np.all(np.isfinite(reaction)):
        raise RuntimeError('BVP evaluation produced non-finite profiles.')
    if np.min(concentrations) < -2e-3:
        raise RuntimeError('Unphysical negative concentrations in the BVP solution.')

    return {'x': x, 'concentrations': concentrations,
            'reaction': reaction, 'Da': Da,
            'min_concentration': float(np.min(concentrations)),
            'n_nodes': sol.x.size}


def profile_for_da(Da):
    '''Use actual Da values (including zero); rounding makes the cache reusable.'''
    if not np.isfinite(Da) or Da < 0:
        raise ValueError('Da must be finite and nonnegative.')
    key = -np.inf if Da == 0.0 else round(float(np.log10(Da)), 9)
    return _profile_for_key(key)


def profile_for_log_da(log_da):
    '''Compatibility helper for an existing log10(Da) value.'''
    return profile_for_da(10.0**log_da)


#%% Two vertically stacked plots with shared x-axis
def make_figure(bottom=0.18, title='Electroneutral ion transport'):
    '''Two panels only: concentrations above reaction, identical x domain.'''
    import matplotlib.pyplot as plt
    plt.rcParams.update(plot_style)
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.2), dpi=110,
                             sharex=True, gridspec_kw={'height_ratios': [1.35, 1.0]})
    fig.subplots_adjust(left=0.13, right=0.94, top=0.88,
                        bottom=bottom, hspace=0.23)
    fig.suptitle(title, fontsize=14)

    names = [r'$H^+$', r'$Na^+$', r'$OH^-$', r'$Cl^-$']
    palette = [colors['H'], colors['Na'], colors['OH'], colors['Cl']]
    lines = [axes[0].plot([], [], lw=2.0, color=color, label=name)[0]
             for name, color in zip(names, palette)]
    reaction_line, = axes[1].plot([], [], lw=2.2, color=colors['R'])

    axes[0].set_title('Ionic concentration profiles', fontsize=12)
    axes[1].set_title('Volumetric reaction-rate profile', fontsize=12)
    axes[0].legend(loc='upper right', ncol=2, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    status = fig.text(0.13, 0.015, '', fontsize=9, va='bottom')
    return fig, axes, lines, reaction_line, status


def plot_profiles(axes, lines, reaction_line, profile, dimensional=False,
                  C_star=1.0, L=1.0, D_star=1.0):
    '''Redraw profiles with common x-axis (using the selected unit system).'''
    x = profile['x']
    c = profile['concentrations']
    r = profile['reaction']

    if dimensional:
        x = x * L * 1e6                 # m -> micrometers
        c = c * C_star                  # mol/m^3
        r = r * D_star * C_star / L**2   # mol/(m^3 s)
        axes[0].set_ylabel(r'$c_i$ (mol m$^{-3}$)')
        axes[1].set_ylabel(r'$R$ (mol m$^{-3}$ s$^{-1}$)')
        axes[1].set_xlabel(r'$x$ ($\mu$m)')
    else:
        axes[0].set_ylabel(r'$\widehat{C}_i$')
        axes[1].set_ylabel(r'$\widehat{R}$')
        axes[1].set_xlabel(r'$\widehat{x}$')

    for line, values in zip(lines, c):
        line.set_data(x, values)
    reaction_line.set_data(x, r)

    # sharex=True synchronizes x-limits across both axes.
    axes[1].set_xlim(float(x[0]), float(x[-1]))
    axes[0].set_ylim(0.0, max(1e-12, 1.15 * float(np.max(c))))
    axes[1].set_ylim(0.0, max(1e-12, 1.15 * float(np.max(r))))
