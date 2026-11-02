# -*- coding: utf-8 -*-
'''
DIMENSIONAL steady electroneutral ion transport: responsive five-slider app.

Concentrations and reaction-rate profiles appear in vertically stacked plots
with a common x-axis. Sliders update while being dragged, without new BVP
solves; a precomputed 2D library depends on Da and gamma=cs/c0.

Five dimensional parameters:
  c0, cs  : 0.05 to 2.0 mol/m^3, both LINEAR
  L       : 1e-9 to 1e0 m, LOGARITHMIC
  k       : 0 and 1e-12 to 1e10 m^3/(mol s), LOGARITHMIC + zero step
  D_star  : 1e-9 to 1e-1 m^2/s, LOGARITHMIC

This is an INTERPOLATED NUMERICAL DEMONSTRATION. The library contains exact
SciPy solve_bvp snapshots, but intermediate settings use interpolation.
For research/quantitative comparisons, solve the BVP at the exact parameters.

Run this file alongside en_slider_core.py, en_dimensional_lookup.py, and
'en_dimensional_profiles_2d.npz'. All relative D_i/D_star remain fixed.
'''

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from en_slider_core import c0, cs, L, D_star_ref, colors, make_figure, make_plots
from en_dimensional_lookup import ENProfileLookup

#%% Physical slider ranges
C_MIN, C_MAX = 0.05, 2.0
LOG_L_MIN, LOG_L_MAX = -9.0, 0.0
LOG_D_MIN, LOG_D_MAX = -9.0, -1.0

# One special left-hand step corresponds to k=0; otherwise k=10^q.
LOG_K_OFF = -12.25
LOG_K_MIN, LOG_K_MAX = -12.0, 10.0
LOG_K_STEP = 0.25

C0_INITIAL = c0
CS_INITIAL = cs
L_INITIAL = L
K_INITIAL = 1e4
D_INITIAL = D_star_ref

#%% Read pre-solved table (once at launch)
lookup = ENProfileLookup()

#%% Vertically stacked axes, shared x-axis and matching species palette
fig, axes, concentration_lines, reaction_line, status = make_figure(
    bottom=0.47, title='Dimensional electroneutral ion transport'
)
fig.set_size_inches(9.0, 9.0)
fig.subplots_adjust(left=0.15, right=0.93, top=0.91, bottom=0.47, hspace=0.24)
axes[0].set_ylabel(r'$c_i$ (mol m$^{-3}$)')
axes[1].set_ylabel(r'$R$ (mol m$^{-3}$ s$^{-1}$)')

# Localize the reaction-rate maximum, like the old dimensional code.
peak_line, = axes[1].plot([], [], '--', color='0.35', lw=1.15)
peak_point, = axes[1].plot([], [], 'o', color=colors['R'], ms=4)
peak_text = axes[1].text(0.02, 0.96, '', transform=axes[1].transAxes,
    va='top', ha='left', fontsize=9,
    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.90, 'pad': 5})


def choose_x_unit(length):
    '''Use convenient units while always showing the same x on both plots.'''
    if length < 1e-6:
        return 1e9, 'nm'
    if length < 1e-3:
        return 1e6, r'$\mu$m'
    if length < 1.0:
        return 1e3, 'mm'
    return 1.0, 'm'


#%% Five sliders: concentrations linear, L/k/D* logarithmic
labels = [r'$c_0$ (mol m$^{-3}$)', r'$c_s$ (mol m$^{-3}$)',
          r'$L$', r'$k$', r'$D_*$']
ranges = [
    (C_MIN, C_MAX, C0_INITIAL, 0.01),
    (C_MIN, C_MAX, CS_INITIAL, 0.01),
    (LOG_L_MIN, LOG_L_MAX, np.log10(L_INITIAL), 0.05),
    (LOG_K_OFF, LOG_K_MAX, np.log10(K_INITIAL), LOG_K_STEP),
    (LOG_D_MIN, LOG_D_MAX, np.log10(D_INITIAL), 0.05)
]
slider_ys = [0.384, 0.318, 0.252, 0.186, 0.120]
sliders = []
for label, (minimum, maximum, initial, step), ypos in zip(labels, ranges, slider_ys):
    ax_slider = fig.add_axes([0.39, ypos, 0.45, 0.027])
    sliders.append(Slider(ax_slider, label, minimum, maximum,
                          valinit=initial, valstep=step, dragging=True))


def physical_values():
    '''Convert slider coordinates to physical quantities.'''
    c0_value = sliders[0].val
    cs_value = sliders[1].val
    L_value = 10.0**sliders[2].val
    log_k = sliders[3].val
    k_value = 0.0 if log_k < LOG_K_MIN else 10.0**log_k
    D_value = 10.0**sliders[4].val
    return c0_value, cs_value, L_value, k_value, D_value


def show_values(c0_value, cs_value, L_value, k_value, D_value):
    '''Display actual dimensional values instead of base-10 logarithms.'''
    for slider, label in zip(sliders, [f'{c0_value:.2f}', f'{cs_value:.2f}',
                                      f'{L_value:.2e} m', f'{k_value:.2e}',
                                      f'{D_value:.2e}']):
        slider.valtext.set_text(label)


def clear_profile():
    '''Do not leave an outdated numerical result visible after an error.'''
    for line in concentration_lines:
        line.set_data([], [])
    reaction_line.set_data([], [])
    peak_line.set_data([], [])
    peak_point.set_data([], [])
    peak_text.set_text('')


def update(_=None):
    '''Fast update using the 2D BVP table; NO solve_bvp call on mouse move.'''
    c0_value, cs_value, L_value, k_value, D_value = physical_values()
    Da = k_value * c0_value * L_value**2 / D_value
    gamma = cs_value / c0_value
    show_values(c0_value, cs_value, L_value, k_value, D_value)

    try:
        profile = lookup.concentrations(Da, gamma)
        xhat, c_hat = profile['x'], profile['concentrations']
        multiplier, unit = choose_x_unit(L_value)
        x = xhat * L_value * multiplier
        c = c_hat * c0_value
        r = k_value * c[0] * c[2]   # EXACT dimensional reaction law
        # Equivalently, r = (D_star*c0/L**2)*Da*C_H*C_OH.

        for line, values in zip(concentration_lines, c):
            line.set_data(x, values)
        reaction_line.set_data(x, r)

        i_peak = int(np.argmax(r))
        x_peak, r_peak = float(x[i_peak]), float(r[i_peak])
        if r_peak > 0.0:
            peak_line.set_data([x_peak, x_peak], [0.0, r_peak])
            peak_point.set_data([x_peak], [r_peak])
            peak_text.set_text(f'$Da$ = {Da:.2e}\nPeak x = {x_peak:.3g}')
        else:
            peak_line.set_data([], [])
            peak_point.set_data([], [])
            peak_text.set_text('$Da=0$: no reaction')

        axes[1].set_xlim(0.0, L_value * multiplier)
        axes[0].set_ylim(0.0, max(1e-12, 1.15*float(np.max(c))))
        axes[1].set_ylim(0.0, max(1e-12, 1.15*float(np.max(r))))
        axes[1].set_xlabel(r'$x$ (' + unit + ')')

        status.set_text(f'Da={Da:.3e}    cs/c0={gamma:.3g}    '
            f'L={L_value:.2e} m    k={k_value:.2e}    D*={D_value:.2e} m²/s\n'
            'Smooth preview: interpolated from precomputed BVP solutions')
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        clear_profile()
        status.set_text(f'Unable to show validated preview: {exc}')
    fig.canvas.draw_idle()


for slider in sliders:
    slider.on_changed(update)

update()
if make_plots:
    plt.show()
