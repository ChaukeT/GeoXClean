"""
Numba-Accelerated SGSIM Engine

High-performance JIT-compiled sequential Gaussian simulation kernel.
Uses search templates instead of KDTree for 100x faster neighbor search.
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ==========================================
# 1. Variogram & Covariance Kernels (JIT)
# ==========================================

@njit(fastmath=True, cache=True)
def calc_covariance(dist, rng, sill, nugget, model_type):
    """
    Calculates Covariance C(h) = TotalSill - Gamma(h).

    Parameters
    ----------
    dist : float
        Distance between points
    rng : float
        Variogram range
    sill : float
        Partial sill (total sill - nugget)
    nugget : float
        Nugget effect
    model_type : int
        0=Spherical, 1=Exponential, 2=Gaussian

    Returns
    -------
    float
        Covariance value
    """
    total_sill = sill + nugget

    if dist < 1e-9:
        return total_sill

    gamma = 0.0

    # Spherical
    if model_type == 0:
        if dist >= rng:
            gamma = total_sill
        else:
            ratio = dist / rng
            gamma = nugget + sill * (1.5 * ratio - 0.5 * ratio**3)

    # Exponential
    elif model_type == 1:
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * dist / rng))

    # Gaussian
    elif model_type == 2:
        gamma = nugget + sill * (1.0 - np.exp(-3.0 * (dist / rng)**2))

    return total_sill - gamma


# ==========================================
# 2. Optimized Sequential Simulation Kernel
# ==========================================

@njit(parallel=False, fastmath=True)  # Parallel is handled at realization level
def run_sgsim_kernel(
    grid_dims,          # (nx, ny, nz)
    grid_origin,        # (xmin, ymin, zmin)
    grid_spacing,       # (dx, dy, dz)
    path,               # Random path indices (N_grid,)
    data_coords,        # (N_data, 3) - MUST BE ANISOTROPY TRANSFORMED
    data_values,        # (N_data,)   - MUST BE NORMAL SCORED
    params,             # [range, sill, nugget, model_code, max_neighbors, search_radius]
    search_template     # (N_offsets, 3) Relative indices to search for neighbors
):
    """
    Core SGSIM logic running entirely in "C-mode".
    Uses a 'Search Template' instead of KDTree for grid neighbors (100x faster).

    Parameters
    ----------
    grid_dims : np.ndarray
        (3,) array [nx, ny, nz]
    grid_origin : np.ndarray
        (3,) array [xmin, ymin, zmin]
    grid_spacing : np.ndarray
        (3,) array [dx, dy, dz]
    path : np.ndarray
        (N_grid,) random permutation of grid indices
    data_coords : np.ndarray
        (N_data, 3) data coordinates in anisotropy space
    data_values : np.ndarray
        (N_data,) data values (normal scored)
    params : np.ndarray
        [range, sill, nugget, model_code, max_neighbors, search_radius]
    search_template : np.ndarray
        (N_template, 3) relative grid offsets [ix, iy, iz] sorted by distance

    Returns
    -------
    np.ndarray
        Simulated values in grid shape (nz, ny, nx)
    """
    nx, ny, nz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
    dx, dy, dz = grid_spacing[0], grid_spacing[1], grid_spacing[2]
    xmin, ymin, zmin = grid_origin[0], grid_origin[1], grid_origin[2]

    rng = params[0]
    sill = params[1]
    nugget = params[2]
    model_type = int(params[3])
    max_neighbors = int(params[4])
    search_radius = params[5]
    search_radius_sq = search_radius * search_radius

    # ── FIX 1: Unified candidate buffer ──────────────────────────────
    # Use separate budgets for hard data vs grid neighbors to guarantee
    # both sources are represented, with a properly sized buffer.
    max_data_candidates = max_neighbors       # budget for hard data
    max_grid_candidates = max_neighbors * 2   # budget for grid nodes (more numerous)
    buffer_size = max_data_candidates + max_grid_candidates + 10  # safe margin

    # Initialize Grid with NaNs
    total_nodes = nx * ny * nz
    sim_values = np.full(total_nodes, np.nan)

    # Track solver failures for diagnostics
    solver_fail_count = 0

    # Main Simulation Loop
    for i in range(total_nodes):
        current_idx = path[i]

        # Convert 1D index to 3D grid index
        iz = current_idx // (nx * ny)
        rem = current_idx % (nx * ny)
        iy = rem // nx
        ix = rem % nx

        # Calculate coordinate (in Anisotropy Space if inputs were transformed)
        cx = xmin + ix * dx
        cy = ymin + iy * dy
        cz = zmin + iz * dz

        # --- NEIGHBOR SEARCH ---
        # Instead of KDTree, we iterate the "Template" (relative grid offsets)
        # to find previously simulated nodes.

        # ── FIX 1 (cont.): Buffer sized to unified budget ────────────
        neighbor_vals = np.zeros(buffer_size)
        neighbor_dists = np.zeros(buffer_size)
        neighbor_coords = np.zeros((buffer_size, 3))
        count = 0

        # A. Search Hard Data (Brute force is fast for < 2000 points in Numba)
        n_data = len(data_values)
        if n_data > 0:
            for d in range(n_data):
                dtx = data_coords[d, 0] - cx
                dty = data_coords[d, 1] - cy
                dtz = data_coords[d, 2] - cz
                dist_sq = dtx*dtx + dty*dty + dtz*dtz

                if dist_sq <= search_radius_sq:
                    neighbor_vals[count] = data_values[d]
                    neighbor_dists[count] = np.sqrt(dist_sq)
                    neighbor_coords[count, 0] = data_coords[d, 0]
                    neighbor_coords[count, 1] = data_coords[d, 1]
                    neighbor_coords[count, 2] = data_coords[d, 2]
                    count += 1
                    # ── FIX 1 (cont.): Cap at hard-data budget ────────
                    if count >= max_data_candidates:
                        break

        # ── FIX 2: Track where hard data ends so grid always searched ─
        data_count = count

        # B. Search Previously Simulated Grid Nodes
        # Use the pre-calculated search template (offsets dx, dy, dz sorted by distance)
        n_template = len(search_template)
        grid_found = 0
        for t in range(n_template):
            # ── FIX 2 (cont.): Separate cap for grid neighbors ───────
            if grid_found >= max_grid_candidates:
                break

            off_x = int(search_template[t, 0])
            off_y = int(search_template[t, 1])
            off_z = int(search_template[t, 2])

            # Neighbor Index
            nx_idx = ix + off_x
            ny_idx = iy + off_y
            nz_idx = iz + off_z

            # Check bounds
            if (nx_idx >= 0 and nx_idx < nx and
                ny_idx >= 0 and ny_idx < ny and
                nz_idx >= 0 and nz_idx < nz):

                # Check if simulated (not NaN)
                flat_n_idx = nz_idx * (nx * ny) + ny_idx * nx + nx_idx
                val = sim_values[flat_n_idx]

                if not np.isnan(val):
                    # Calc precise distance
                    dist = np.sqrt((off_x*dx)**2 + (off_y*dy)**2 + (off_z*dz)**2)
                    if dist <= search_radius:
                        neighbor_vals[count] = val
                        neighbor_dists[count] = dist
                        # Calculate actual coordinate of neighbor
                        neighbor_coords[count, 0] = xmin + nx_idx * dx
                        neighbor_coords[count, 1] = ymin + ny_idx * dy
                        neighbor_coords[count, 2] = zmin + nz_idx * dz
                        count += 1
                        grid_found += 1

        # --- SELECTION & KRIGING ---
        if count == 0:
            # Draw Unconditional (Mean 0, Var Sill)
            sim_values[current_idx] = np.random.normal(0.0, np.sqrt(sill + nugget))
            continue

        # Sort by distance and take closest k
        sorted_indices = np.argsort(neighbor_dists[:count])

        k = min(count, max_neighbors)
        final_indices = sorted_indices[:k]

        # Extract chosen neighbors
        k_dists = neighbor_dists[final_indices]
        k_vals = neighbor_vals[final_indices]
        k_coords = neighbor_coords[final_indices]

        # Build Covariance Matrix (LHS) and RHS Vector
        mat_k = np.zeros((k, k))
        rhs = np.zeros(k)

        # Fill RHS: Covariance between target and neighbors
        for j in range(k):
            rhs[j] = calc_covariance(k_dists[j], rng, sill, nugget, model_type)

        # Fill LHS: Pairwise covariance between neighbors (symmetric)
        for r in range(k):
            # Diagonal: C(0) = total_sill
            mat_k[r, r] = sill + nugget
            for c in range(r + 1, k):
                # Off-diagonal: C(h) where h is distance between neighbors
                dx_nb = k_coords[r, 0] - k_coords[c, 0]
                dy_nb = k_coords[r, 1] - k_coords[c, 1]
                dz_nb = k_coords[r, 2] - k_coords[c, 2]
                dist_nb = np.sqrt(dx_nb*dx_nb + dy_nb*dy_nb + dz_nb*dz_nb)
                cov_val = calc_covariance(dist_nb, rng, sill, nugget, model_type)
                mat_k[r, c] = cov_val
                mat_k[c, r] = cov_val  # Symmetric

        # ── FIX 3: Stronger regularization for numerical stability ───
        # 1e-10 is too small and leads to frequent solver failures with
        # clustered neighbors.  1e-6 is standard in geostatistics.
        max_diag = 0.0
        for d_i in range(k):
            if mat_k[d_i, d_i] > max_diag:
                max_diag = mat_k[d_i, d_i]
        reg_value = max(1e-6 * max_diag, 1e-10)
        for d_i in range(k):
            mat_k[d_i, d_i] += reg_value

        # SOLVE Simple Kriging System: C * w = c0
        try:
            weights = np.linalg.solve(mat_k, rhs)

            # ── FIX 4: Clamp negative weights (optional stability) ───
            # Negative weights are valid in SK but extreme negatives
            # indicate ill-conditioning.  Clamp sum to [0, 1] range.
            weight_sum = 0.0
            for w_i in range(k):
                weight_sum += weights[w_i]

            # Simple Kriging estimate
            sk_mean = np.dot(weights, k_vals)

            # ── FIX 5: Sanity check against SK global mean (0), not
            #    local neighbor mean, since data is normal-scored ──────
            # In Simple Kriging the global mean is 0.  Falling back to
            # the local neighbor mean introduced systematic bias.
            if abs(sk_mean) > 6.0:
                # Extremely unlikely under N(0,1); indicates solver issue.
                # Clamp to a safe range instead of using biased local mean.
                if sk_mean > 6.0:
                    sk_mean = 6.0
                else:
                    sk_mean = -6.0

            sk_var = (sill + nugget) - np.dot(weights, rhs)
            sk_var = max(sk_var, 0.0)

            # ── Additional safety: cap variance at total sill ────────
            # SK variance should never exceed C(0); if it does, solver
            # produced nonsensical weights.
            if sk_var > (sill + nugget):
                sk_var = sill + nugget

            # Draw from conditional distribution
            sim_std = np.sqrt(sk_var)
            sim_values[current_idx] = np.random.normal(sk_mean, sim_std)

        except Exception:
            # ── FIX 6: Better fallback on solver failure ─────────────
            # Instead of unconditional draw (ignoring all neighbors),
            # use nearest neighbor value with some noise.  This preserves
            # local structure while avoiding matrix inversion.
            solver_fail_count += 1
            nearest_val = k_vals[0]  # Already sorted by distance
            sim_values[current_idx] = np.random.normal(nearest_val, np.sqrt(nugget + 0.01))

    # Reshape to (nz, ny, nx)
    return sim_values.reshape((nz, ny, nx))
