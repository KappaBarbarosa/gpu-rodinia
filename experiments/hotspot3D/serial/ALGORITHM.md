# hotspot3D — Algorithm Specification

## Overview

hotspot3D simulates 3-D chip thermal dynamics using an **explicit finite-difference
7-point stencil** over a `nx × ny × nz` grid (square cross-section: nx = ny).
Each cell holds a temperature value; each time step updates every cell using its
6 immediate neighbours (W, E, N, S, B, T) plus an external power source and
ambient cooling term. The algorithm is a faithful implementation of Rodinia's
`openmp/hotspot3D/3D.c`, `computeTempCPU`.

## CLI

```
<program> <rows/cols> <layers> <iterations> <output_file>
```

| Argument | Type | Meaning |
|---|---|---|
| `rows/cols` | int | Grid dimension; **nx = ny = rows/cols** (square cross-section) |
| `layers` | int | Z-dimension: nz = layers |
| `iterations` | int | Number of time steps |
| `output_file` | path | Path for the output temperature file |

Example: `./hotspot3D 512 8 100 out.txt`

## Physical Parameters (compile-time constants — do NOT change)

```c
float t_chip      = 0.0005f;    // chip thickness [m]
float chip_height = 0.016f;     // chip height [m]
float chip_width  = 0.016f;     // chip width  [m]
float amb_temp    = 80.0f;      // ambient temperature [°C]

#define MAX_PD        3.0e6f    // used only for dt calculation
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f
```

## Derived Coefficients (from physical parameters + grid size)

```c
float dx  = chip_height / ny;
float dy  = chip_width  / nx;
float dz  = t_chip      / nz;
float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
float Rx  = dy / (2.0f * K_SI * t_chip * dx);
float Ry  = dx / (2.0f * K_SI * t_chip * dy);
float Rz  = dz / (K_SI * dx * dy);
float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
float dt  = PRECISION / max_slope;

float stepDivCap = dt / Cap;
float ce = cw = stepDivCap / Rx;
float cn = cs = stepDivCap / Ry;
float ct = cb = stepDivCap / Rz;
float cc = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
float dtCap = dt / Cap;          // = stepDivCap
```

For the standard workload (nx=ny=512, nz=8):
- ce=cw=cn=cs ≈ 0.034, ct=cb ≈ 5.3e-4, cc ≈ 0.862, dtCap ≈ 0.341 K/step/W

## Input Generation

The original Rodinia data files (`power_512x8`, `temp_512x8`) are no longer
available. Input is generated **deterministically** from `srand(7)`:

```c
srand(7);
// Loop order: i=row (outer), j=col (middle), k=layer (inner)
// Storage index: idx = j + i*nx + k*nx*ny  (i.e. x + y*nx + z*nx*ny)
for (int i = 0; i < ny; i++)
    for (int j = 0; j < nx; j++)
        for (int k = 0; k < nz; k++) {
            int idx = j + i*nx + k*nx*ny;
            power[idx] = (rand() / (float)RAND_MAX) * 15.0f;   // [0, 15] W per cell
            tIn[idx]   = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;  // [80, 85] °C
        }
```

**Critical:** the loop order and per-cell rand() call count (2 calls per cell,
power first then temperature) must be reproduced exactly to get matching input.
`rand()` uses glibc's default LCG seeded with `srand(7)`.

## Stencil Update (per time step) — verbatim from Rodinia computeTempCPU

For each cell `c = x + y*nx + z*nx*ny` (z-outer loop), compute neighbour indices:

```c
int w = (x == 0)      ? c : c - 1;        // west
int e = (x == nx-1)   ? c : c + 1;        // east
int n = (y == 0)      ? c : c - nx;       // north
int s = (y == ny-1)   ? c : c + nx;       // south
int b = (z == 0)      ? c : c - nx*ny;    // bottom
int t = (z == nz-1)   ? c : c + nx*ny;    // top
```

**Boundary condition:** clamped to self (mirror / no-flux).

Update rule:
```c
tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw
        + tIn[t]*ct + tIn[b]*cb + dtCap*power[c] + ct*amb_temp;
```

This matches `computeTempCPU` in `openmp/hotspot3D/3D.c` exactly.

## Time Stepping (ping-pong buffers)

```c
float *cur = tIn, *nxt = tOut;
for (int iter = 0; iter < niter; iter++) {
    // compute stencil: read from cur, write to nxt
    float *tmp = cur; cur = nxt; nxt = tmp;  // swap
}
// cur now holds the final temperature field
```

After all iterations, the **final result is in `cur`** (the buffer that was most
recently written). For `niter=100` (even), this is the buffer that was
originally `tIn`; for odd `niter`, it is the original `tOut`. Always write
from `cur`, not from a hardcoded `tOut`.

## Output Format

Write every cell in the same loop order as input (i=row outer, j=col middle,
k=layer inner):

```
<index>\t<value>\n
```

`<index>` is a running counter starting at 0. `<value>` is printed with `%g`
format (same as `sprintf(str, "%d\t%g\n", index, value)` in the original code).

Total output lines: nx × ny × nz (= 2,097,152 for the standard workload).

## Standard Workload

```
nx = ny = 512,  nz = 8,  iterations = 100
```

Output: 2,097,152 lines of float temperature values in [°C].  
Serial compute time: ~0.73 s on a modern CPU.

## Compute Time Reporting

Print to **stderr**:
```
compute_seconds: <elapsed>
```
where `<elapsed>` is the wall time of the stencil loop only (excluding I/O and
input generation), measured with `gettimeofday`.

## Parallelism

The dominant parallelism: **all nx×ny×nz cells within one time step are
independent** (reads from `cur`, writes to `nxt`). This is a 3-D data-parallel
kernel with a sequential dependency between time steps.

Phase-2 optimization targets: shared-memory tiling in the xy-plane (or xyz
block), temporal tiling (process multiple time steps per block to amortise
global memory traffic), float4 vectorization along x, loop coalescing.
