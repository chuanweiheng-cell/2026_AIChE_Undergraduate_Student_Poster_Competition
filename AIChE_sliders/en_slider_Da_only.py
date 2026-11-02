# -*- coding: utf-8 -*-
'''
INSTANT nondimensional electroneutral slider, Da = 10**q.

The 321 profiles in en_da_profiles.npz are genuine SciPy solve_bvp solutions,
precomputed at q increments of 0.05 over [-6, 10]. Slider motion only
selects a saved solution; it never reruns a BVP, so no solver wait is needed.

If you change the model or numerical settings in en_slider_core.py,
rerun en_slider_precompute.py to rebuild the lookup table.
'''

#%% Imports
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from en_slider_core import make_figure, make_plots, plot_profiles
from en_slider_precompute import (ProfileTable, CACHE_PATH, make_table,
                                  LOG_DA_MIN, LOG_DA_MAX, LOG_DA_STEP)

#%% Load precomputed profiles (generate once if not yet available)
try:
    profiles = ProfileTable()
except (FileNotFoundError, ValueError) as exc:
    print(f'{exc}\nBuilding one-time BVP lookup table...')
    make_table(CACHE_PATH)
    profiles = ProfileTable()

#%% Plots and a single logarithmic Damkohler slider
fig, axes, concentration_lines, reaction_line, status = make_figure(
    bottom=0.23, title='Nondimensional electroneutral ion transport')
slider_ax = fig.add_axes([0.24, 0.104, 0.63, 0.032])

# The slider uses q=log10(Da); the displayed number is Da itself.
da_slider = Slider(slider_ax, r'$Da$', LOG_DA_MIN, LOG_DA_MAX,
                   valinit=2.0, valstep=LOG_DA_STEP, dragging=True)


def update(_=None):
    '''Instantly retrieve and display a previously solved ENNP profile.'''
    profile = profiles.at_log_da(da_slider.val)
    Da = profile['Da']
    da_slider.valtext.set_text(f'{Da:.2e}')
    plot_profiles(axes, concentration_lines, reaction_line, profile)
    status.set_text(f'Da = {Da:.4g}   Precomputed BVP (no solving during drag)')
    fig.canvas.draw_idle()


da_slider.on_changed(update)
update()
if make_plots:
    plt.show()
