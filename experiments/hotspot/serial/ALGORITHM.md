# HotSpot — Algorithm Specification

## Problem
Transient thermal simulation of a 2D chip. The chip is modelled as a
`grid_rows × grid_cols` grid of cells. Each cell has a temperature and a
constant power-dissipation value. Starting from an initial temperature grid,
the simulation advances the temperature field by `sim_time` discrete time
steps and writes the final temperatures.

## Governing update (one time step)
For every cell `(r, c)` the new temperature is the old temperature plus a
`delta` computed from a 5-point stencil over the **previous** time step's
temperatures, the cell's own power, and a coupling to a fixed ambient
temperature:

```
delta(r,c) = Cap_1 * ( power(r,c)
                     + (T_left + T_right  - 2*T_center) * Rx_1     // horizontal conduction
                     + (T_up   + T_down   - 2*T_center) * Ry_1     // vertical conduction
                     + (amb_temp          -   T_center) * Rz_1 )   // conduction to ambient
new_T(r,c) = T_center + delta(r,c)
```

`T_center = temp(r,c)`, and `T_left/T_right/T_up/T_down` are the four orthogonal
neighbours. The coefficients `Cap_1, Rx_1, Ry_1, Rz_1` are scalars derived once
from the chip's physical parameters (see `compute_tran_temp`).

## Boundary conditions
Cells on the border have fewer than four neighbours. The reference handles the
four **corners**, the four **edges**, and the **interior** as separate cases
(see `single_iteration` in `hotspot_serial.c`). At a missing neighbour the
corresponding conduction term simply drops (a one-sided difference is used
instead of the centered `(left+right-2*center)` form). The exact per-case
expressions in the reference are normative — reproduce them precisely.

## Key parallelism property
Within a single time step, **every output cell depends only on the previous
step's temperature grid** — never on another cell's *new* value. Therefore all
`grid_rows × grid_cols` cell updates of one iteration are mutually independent
and can be computed in any order / fully in parallel. Iterations themselves are
sequential: step `i+1` reads the temperatures produced by step `i`. The
reference uses two buffers and ping-pongs them between iterations.

## I/O contract (must match exactly)
Command line:
```
<prog> <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>
```
- `temp_file` / `power_file`: text, one float per line, `grid_rows*grid_cols`
  lines, row-major.
- `output_file`: text, one line per cell, `"<index>\t<temperature>\n"`,
  row-major, printed with C `"%g"`. After an even number of iterations the
  final data is in the original temperature buffer; after an odd number it is
  in the result buffer (the reference selects `(sim_time & 1) ? result : temp`).

## Numerical type
All computation is single-precision `float`.
