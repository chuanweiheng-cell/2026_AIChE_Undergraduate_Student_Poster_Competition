# -*- coding: utf-8 -*-
'''
Precompute accurate ENNP BVP solutions for instant Damkohler slider updates.

Run once after changing numerical settings or physical ratios:
    python en_slider_precompute.py

The generated .npz contains the actual SciPy BVP profiles at 321 log-spaced
Damkohler numbers. No interpolation or BVP solving occurs during slider motion.
'''

#%% Imports
from pathlib import Path
import hashlib
import time
import numpy as np
from en_slider_core import (N_MESH, N_PLOT, initial_guess, _solve_one,
                            SALT_RATIO, D_REL)

#%% Slider sampling and saved file
LOG_DA_MIN = -6.0
LOG_DA_MAX = 10.0
LOG_DA_STEP = 0.05
CACHE_PATH = Path(__file__).with_name('en_da_profiles.npz')
CORE_PATH = Path(__file__).with_name('en_slider_core.py')


def signature():
    '''Hash the model source and cache sampling; rebuild if either changes.'''
    data = CORE_PATH.read_bytes()
    data += repr((LOG_DA_MIN, LOG_DA_MAX, LOG_DA_STEP, N_PLOT)).encode()
    return hashlib.sha256(data).hexdigest()


def make_table(save_path=CACHE_PATH):
    '''Continue across Da once; save exact BVP profiles on their refined meshes.'''
    start = time.perf_counter()
    n = round((LOG_DA_MAX - LOG_DA_MIN) / LOG_DA_STEP)
    log_values = np.linspace(LOG_DA_MIN, LOG_DA_MAX, n + 1)

    x0 = np.linspace(0.0, 1.0, N_MESH)
    y0, p0 = initial_guess(x0)
    sol = _solve_one(0.0, x0, y0, p0)

    x_parts, c_parts, r_parts = [], [], []
    offsets = [0]
    bc_max = 0.0
    min_conc = float('inf')

    for i, q in enumerate(log_values):
        Da = 10.0**float(q)
        sol = _solve_one(Da, sol.x, sol.y, sol.p)
        x = np.unique(np.concatenate((np.linspace(0.0, 1.0, N_PLOT), sol.x)))
        cH, cNa, cOH = sol.sol(x)[:3]
        cCl = cH + cNa - cOH
        c = np.stack((cH, cNa, cOH, cCl), axis=0)
        reaction = Da * cH * cOH

        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(reaction)):
            raise RuntimeError(f'Non-finite solution at Da={Da:.6g}')
        if np.min(c) < -2e-3:
            raise RuntimeError(f'Negative concentration at Da={Da:.6g}')

        # Check boundary conditions and the identity cCl = cH + cNa - cOH.
        bc_errors = np.array([cH[0] - 1.0, cNa[0] - SALT_RATIO, cOH[0],
                              cH[-1], cNa[-1] - 1.0, cOH[-1] - 1.0])
        bc_max = max(bc_max, float(np.max(np.abs(bc_errors))))
        min_conc = min(min_conc, float(np.min(c)))

        x_parts.append(x)
        c_parts.append(c)
        r_parts.append(reaction)
        offsets.append(offsets[-1] + x.size)
        if i % 40 == 0 or i == n:
            print(f'  {i + 1:3d}/{n + 1} profiles | Da={Da:.3e} | nodes={sol.x.size}', flush=True)

    # Variable-length mesh arrays are concatenated; offsets retrieve each case.
    # No object arrays or pickle required.
    np.savez_compressed(save_path,
                        log_da=log_values,
                        offsets=np.asarray(offsets, dtype=np.int64),
                        x=np.concatenate(x_parts),
                        concentrations=np.concatenate(c_parts, axis=1),
                        reaction=np.concatenate(r_parts),
                        signature=np.array(signature()),
                        salt_ratio=np.array(SALT_RATIO), d_rel=D_REL)

    print(f'Precompute complete in {time.perf_counter() - start:.2f} s.')
    print(f'Largest boundary error: {bc_max:.2e}; smallest concentration: {min_conc:.2e}')
    print(f'Saved {n+1} solutions to {save_path}')
    return save_path


class ProfileTable:
    '''Read-only lookup. Each update returns a pre-solved physical BVP profile.'''
    def __init__(self, path=CACHE_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f'Missing {path}; run en_slider_precompute.py')
        with np.load(path, allow_pickle=False) as saved:
            if saved['signature'].item() != signature():
                raise ValueError('Outdated ENNP cache: run en_slider_precompute.py')
            self.log_da = saved['log_da'].copy()
            self.offsets = saved['offsets'].copy()
            self.x = saved['x'].copy()
            self.concentrations = saved['concentrations'].copy()
            self.reaction = saved['reaction'].copy()

    def at_log_da(self, log_da):
        '''Nearest table index; slider snaps to exactly this 0.05-decade grid.'''
        q = float(log_da)
        i = int(np.clip(np.rint((q - LOG_DA_MIN) / LOG_DA_STEP),
                        0, self.log_da.size - 1))
        a, b = self.offsets[i:i+2]
        return {'Da': 10.0**float(self.log_da[i]),
                'x': self.x[a:b],
                'concentrations': self.concentrations[:, a:b],
                'reaction': self.reaction[a:b]}


if __name__ == '__main__':
    make_table()
