# -*- coding: utf-8 -*-
'''
Precomputed two-parameter ENNP lookup for the fast five-slider illustration.

The DIMENSIONLESS solution depends only on
    Da = k*c0*L**2 / D_star
    gamma = cs/c0
when all relative diffusivities D_i/D_star stay fixed.

BVP solutions are computed once on a 2D (Da, gamma) grid, retaining adaptive
mesh points. During mouse movement the slider interpolates the stored
concentration profiles; no live numerical solve is performed.

IMPORTANT: Between grid points the displayed profiles are approximations,
not exact BVP solutions. This is intended for teaching and presentations.
Original scipy solve_bvp remains the reference for validation.
'''

#%% Imports
from pathlib import Path
import hashlib
import time
import numpy as np
from scipy.integrate import solve_bvp
from en_slider_core import D_REL, EPS, N_MESH, BVP_TOL, MAX_NODES, ennp_ode

#%% Precomputation settings
LOG_DA_MIN = -10.0
LOG_DA_MAX = 20.0
LOG_DA_STEP = 0.2
N_GAMMA = 25
GAMMA_MIN = 0.025             # cs/c0 at c0=2.0, cs=0.05
GAMMA_MAX = 40.0              # cs/c0 at c0=0.05, cs=2.0

CACHE_PATH = Path(__file__).with_name('en_dimensional_profiles_2d.npz')
CORE_PATH = Path(__file__).with_name('en_slider_core.py')


def grid_parameters():
    '''Dimensionless 2D grid, including the default ratio 0.2 exactly.'''
    nq = round((LOG_DA_MAX - LOG_DA_MIN) / LOG_DA_STEP)
    log_da = np.linspace(LOG_DA_MIN, LOG_DA_MAX, nq + 1)
    gamma = np.geomspace(GAMMA_MIN, GAMMA_MAX, N_GAMMA)
    gamma = np.unique(np.sort(np.append(gamma, 0.2)))
    return log_da, gamma


def signature():
    '''Prevent accidental reuse after changes to the underlying model.'''
    contents = CORE_PATH.read_bytes() + repr((LOG_DA_MIN, LOG_DA_MAX,
        LOG_DA_STEP, N_GAMMA, GAMMA_MIN, GAMMA_MAX)).encode('utf-8')
    return hashlib.sha256(contents).hexdigest()


def bc_for_gamma(gamma):
    '''Six boundary residuals for the five-state/one-parameter ENNP BVP.'''
    def bc(ya, yb, p):
        return np.array([ya[0] - 1.0, ya[1] - gamma, ya[2],
                         yb[0], yb[1] - 1.0, yb[2] - 1.0])
    return bc


def guess_for_gamma(x, gamma):
    '''Guess consistent with the boundary concentrations.'''
    y = np.zeros((5, x.size))
    y[0] = 1.0 - x
    y[1] = gamma + (1.0 - gamma) * x
    y[2] = x
    y[3] = 1.0
    y[4] = -1.0
    p = np.array([gamma - 1.0])
    return y, p


def solve_at(Da, gamma, x, y, p):
    '''A single exact solve on its own adaptively refined BVP mesh.'''
    sol = solve_bvp(lambda xx, yy, pp: ennp_ode(xx, yy, pp, Da),
        bc_for_gamma(gamma), x, y, p=p, tol=BVP_TOL, max_nodes=MAX_NODES)
    if not sol.success:
        raise RuntimeError(f'Da={Da:.3e}, cs/c0={gamma:.4g}: {sol.message}')
    return sol


def make_table(save_path=CACHE_PATH):
    '''Generate the table; this may take a minute or two on a laptop.'''
    started = time.perf_counter()
    log_da_grid, gamma_grid = grid_parameters()
    x_parts, c_parts, peaks, offsets = [], [], [], [0]
    maximum_bc_error = 0.0
    minimum_concentration = float('inf')
    n_total = len(log_da_grid) * len(gamma_grid)
    count = 0

    for gamma in gamma_grid:
        x = np.linspace(0.0, 1.0, N_MESH)
        y, p = guess_for_gamma(x, gamma)
        sol = solve_at(0.0, gamma, x, y, p)

        for log_da in log_da_grid:
            Da = 10.0**float(log_da)
            sol = solve_at(Da, gamma, sol.x, sol.y, sol.p)

            # Midpoints help preserve the increasingly narrow reaction front.
            x = np.unique(np.r_[sol.x, (sol.x[1:] + sol.x[:-1]) * 0.5])
            cH, cNa, cOH = sol.sol(x)[:3]
            concentrations = np.vstack((cH, cNa, cOH))
            reaction = Da * np.maximum(cH, 0.0) * np.maximum(cOH, 0.0)
            peak = float(x[np.argmax(reaction)])

            if not np.all(np.isfinite(concentrations)):
                raise RuntimeError(f'Nonfinite BVP profile at Da={Da}, gamma={gamma}')
            if np.min(concentrations) < -2e-3:
                raise RuntimeError(f'Negative BVP concentration at Da={Da}, gamma={gamma}')

            bcs = [cH[0] - 1.0, cNa[0] - gamma, cOH[0],
                   cH[-1], cNa[-1] - 1.0, cOH[-1] - 1.0]
            maximum_bc_error = max(maximum_bc_error, float(np.max(np.abs(bcs))))
            minimum_concentration = min(minimum_concentration,
                                        float(np.min(concentrations)))

            x_parts.append(x)
            c_parts.append(concentrations.astype(np.float32))
            peaks.append(peak)
            offsets.append(offsets[-1] + x.size)
            count += 1
        print(f'Finished gamma={gamma:.5g}: {count}/{n_total} profiles; '
              f'elapsed {time.perf_counter()-started:.1f}s', flush=True)

    np.savez_compressed(save_path,
        log_da=log_da_grid, gamma=gamma_grid, offsets=np.asarray(offsets),
        x=np.concatenate(x_parts), c=np.concatenate(c_parts, axis=1),
        peak=np.array(peaks), signature=np.array(signature()),
        relative_diffusivities=D_REL)
    print(f'Saved {n_total} numerical BVP profiles in '
          f'{time.perf_counter()-started:.1f}s to {save_path}', flush=True)
    print(f'Max BVP boundary error {maximum_bc_error:.3g}; '
          f'min concentration {minimum_concentration:.3g}', flush=True)
    return save_path


class ENProfileLookup:
    '''Read 2D snapshots and interpolate for smooth, solver-free mouse motion.'''
    def __init__(self, path=CACHE_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing {path}. Run en_dimensional_lookup.py once.')
        with np.load(path, allow_pickle=False) as data:
            if data['signature'].item() != signature():
                raise ValueError('2D BVP table is outdated. Run en_dimensional_lookup.py.')
            self.log_da = data['log_da'].copy()
            self.gamma = data['gamma'].copy()
            self.offsets = data['offsets'].copy()
            self.x = data['x'].copy()
            self.c = data['c'].copy()
            self.peak = data['peak'].copy()
        self.nq = len(self.log_da)

    def get(self, i, j):
        '''Return table snapshot indexed by gamma i and log-Da j.'''
        key = i * self.nq + j
        a, b = self.offsets[key:key + 2]
        return self.x[a:b], self.c[:, a:b], self.peak[key]

    def concentrations(self, Da, gamma):
        '''Interpolate the BVP solutions at the specified Da and salt ratio.

        Front-aligned interpolation avoids artificially producing two reaction
        peaks when a narrow front shifts between neighboring salt ratios.
        The x=0 and x=1 endpoints are kept fixed by piecewise-affine remapping.
        The boundary values are enforced at the end to preserve the selected
        dimensional reservoir conditions exactly.
        '''
        if not np.isfinite(Da) or not np.isfinite(gamma) or Da < 0 or gamma <= 0:
            raise ValueError('Da must be nonnegative and cs/c0 must be positive.')
        if not self.gamma[0] - 1e-12 <= gamma <= self.gamma[-1] + 1e-12:
            raise ValueError('Salt ratio outside precomputed range.')
        q = float(np.log10(Da)) if Da > 0.0 else -np.inf
        if q > self.log_da[-1] + 1e-12:
            raise ValueError('Da is outside the validated precomputed range.')

        # At extremely small Da (including k=0), use the near-zero reaction
        # concentration profile; the plotted reaction is calculated using the
        # *actual* selected k, so it is identically zero when k=0.
        q = float(np.clip(q, self.log_da[0], self.log_da[-1]))
        j0 = int(np.clip(np.searchsorted(self.log_da, q, side='right') - 1,
                         0, self.nq - 2))
        jq = float(np.clip((q - self.log_da[j0]) /
                           (self.log_da[j0 + 1] - self.log_da[j0]), 0.0, 1.0))
        i0 = int(np.clip(np.searchsorted(self.gamma, gamma, side='right') - 1,
                         0, len(self.gamma) - 2))
        # Linear interpolation in gamma (not log gamma) ensures exact cNa(0).
        ig = float(np.clip((gamma - self.gamma[i0]) /
                           (self.gamma[i0 + 1] - self.gamma[i0]), 0.0, 1.0))

        weights = [(i0, j0, (1.0-ig)*(1.0-jq)),
                   (i0, j0+1, (1.0-ig)*jq),
                   (i0+1, j0, ig*(1.0-jq)),
                   (i0+1, j0+1, ig*jq)]
        snapshots = [(self.get(i, j), weight) for i, j, weight in weights
                     if weight > 1e-10]

        # Align changing reaction-front positions before interpolating.
        target_peak = sum(weight * record[2] for record, weight in snapshots)
        target_peak = float(np.clip(target_peak, 1e-8, 1.0-1e-8))
        aligned = []
        for (x, c, source_peak), weight in snapshots:
            s = float(np.clip(source_peak, 1e-8, 1.0-1e-8))
            xx = np.where(x <= s, x * target_peak / s,
                          target_peak + (x-s)*(1.0-target_peak)/(1.0-s))
            aligned.append((xx, c, weight))

        # Include adaptively placed nodes from the four contributing BVPs,
        # plus a regular grid for readability away from the reaction front.
        x_eval = np.unique(np.concatenate([np.linspace(0.0, 1.0, 360)] +
                                          [item[0] for item in aligned]))
        c_eval = np.zeros((3, x_eval.size))
        for x, c, weight in aligned:
            for species in range(3):
                c_eval[species] += weight * np.interp(x_eval, x, c[species])

        # Respect the exact boundary reservoirs, even for off-grid gamma.
        c_eval = np.maximum(c_eval, 0.0)
        c_eval[0, 0], c_eval[1, 0], c_eval[2, 0] = 1.0, gamma, 0.0
        c_eval[0, -1], c_eval[1, -1], c_eval[2, -1] = 0.0, 1.0, 1.0
        # Enforce nonnegative chloride without violating electroneutrality.
        c_eval[2] = np.minimum(c_eval[2], c_eval[0] + c_eval[1])
        c_cl = c_eval[0] + c_eval[1] - c_eval[2]
        concentrations = np.vstack((c_eval, c_cl))
        return {'x': x_eval, 'concentrations': concentrations,
                'Da': Da, 'gamma': gamma, 'interpolated': True,
                'sample_count': len(snapshots)}


if __name__ == '__main__':
    make_table()
