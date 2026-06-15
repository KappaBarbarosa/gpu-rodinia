/*
 * hotspot3D — Phase 2 round-2 optimised CUDA kernel.
 *
 * Starting point: p2r1 (6.571 ms / 100 iters, 113.4x speedup).
 *
 * Round-2 strategy: cooperative 2-step temporal blocking.
 *
 * MOTIVATION
 * ----------
 * The r1 kernel is ~90 % memory-bandwidth bound (≈ 422 GB/s out of 448 GB/s peak
 * on the RTX 3070).  The only way to go significantly faster is to reduce the
 * volume of global memory traffic.
 *
 * The standard register Z-pipeline (vb/vc/vt) already avoids re-reading the
 * interior Z-column from DRAM.  The remaining global reads per block per step are:
 *   • Interior column:  256 loads  (Z-lookahead pipeline)
 *   • XY halo ring:      80 loads  (north/south/east/west neighbours)
 *   • Power array:      256 loads
 *   • Write nxt[]:      256 stores
 * (all × 8 Z-layers) = 6 784 floats/block/step.
 *
 * KEY INSIGHT: if we run TWO time steps inside a single kernel launch, the
 * interior cells computed in step 1 can be KEPT IN REGISTERS and re-used as
 * the "cur" data for step 2, avoiding 2 048 interior loads per block for step 2.
 * Only the XY halo (80 loads × 8 Z = 640 floats/block) still needs to be read
 * from global memory for step 2 (from the step-1 output buffer).
 *
 * With a CUDA Cooperative Groups grid-wide barrier between the two phases, all
 * blocks finish step 1 before any block begins step 2, so the halo reads in
 * phase 2 see the fully-written step-1 output.
 *
 * MEMORY TRAFFIC COMPARISON
 * -------------------------
 * 2 × separate 1-step launches:  2 × 6 784 = 13 568 floats / block / 2 steps
 * 1 cooperative 2-step launch:
 *   Phase 1 (cur → nxt):   6 784 floats
 *   Phase 2 (nxt[halo] + regs → cur):  (640 + 2048) reads + 2048 writes = 4 736 floats
 *   Total:                 11 520 floats / block / 2 steps
 * Savings: 15.1 % reduction in global memory traffic → expected ~15 % faster.
 *
 * BUFFER MANAGEMENT (2 buffers, no 3rd allocation needed)
 * -------------------------------------------------------
 * Phase 1: reads d_cur[], writes d_nxt[].
 * Phase 2: reads d_nxt[] (halo only), uses registers (interior), writes d_cur[].
 * After the cooperative call, d_cur contains the t+2 result.
 * The host never swaps pointers; d_cur and d_nxt stay fixed for all 50 calls.
 *
 * OTHER CHANGES vs r1
 * -------------------
 * • step1_z[NZ] register array: saves step-1 outputs for all NZ Z-layers.
 * • Kernel templated on NZ=8 (avoids branch-prediction cost of z-boundary
 *   checks; compiler constant-folds z==0 and z==NZ-1 per loop iteration when
 *   the loop is partially unrolled or the template arg is concrete).
 * • __ldg() on all global reads for both cur[] and power[] to explicitly route
 *   through the read-only cache.
 * • Symmetric coefficients remain separate (d_cn/d_cs, d_ce/d_cw) to preserve
 *   the exact floating-point order of the serial reference (bit-exact output).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/* ---- physical constants ---- */
#define MAX_PD        3.0e6f
#define PRECISION     0.001f
#define SPEC_HEAT_SI  1.75e6f
#define K_SI          100.0f
#define FACTOR_CHIP   0.5f

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

/* ---- stencil coefficients in constant memory ---- */
__constant__ float d_cc, d_cn, d_cs, d_ce, d_cw, d_ct, d_cb, d_dtCap, d_ambct;

/* ---- tile dimensions ---- */
#define BX  32
#define BY  8
#define SX  (BX + 2)   /* 34: interior + west/east halo */
#define SY  (BY + 2)   /* 10: interior + north/south halo */

/*
 * Two-step cooperative stencil kernel, templated on NZ.
 *
 * Phase 1: reads cur[], writes nxt[], saves interior outputs in step1_z[NZ].
 * grid.sync()
 * Phase 2: reads nxt[] (halo only via __ldg), uses step1_z[] for interior,
 *           writes cur[] (= the t+2 result).
 *
 * Grid:  ceil(nx/BX) × ceil(ny/BY)
 * Block: BX × BY (= 32 × 8 = 256 threads)
 */
template <int NZ>
__global__ void stencil_2step(
        float       * __restrict__ cur,   /* in: t;   out: t+2 */
        float       * __restrict__ nxt,   /* out: t+1 (intermediate) */
        const float * __restrict__ power,
        int nx, int ny)
{
    cg::grid_group grid = cg::this_grid();

    __shared__ float s[SY][SX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * BX + tx;
    const int gy = blockIdx.y * BY + ty;
    const int nxny = nx * ny;
    const bool valid = (gx < nx) && (gy < ny);

    const int col = gy * nx + gx;

    /* Clamped in-plane neighbour offsets (computed once). */
    const int wcol = (gx > 0)      ? col - 1  : col;
    const int ecol = (gx < nx - 1) ? col + 1  : col;
    const int ncol = (gy > 0)      ? col - nx : col;
    const int scol = (gy < ny - 1) ? col + nx : col;

    const int sx = tx + 1;
    const int sy = ty + 1;

    /* Halo-ring edge flags. */
    const bool wEdge = (tx == 0);
    const bool eEdge = (tx == BX - 1) || (gx == nx - 1);
    const bool nEdge = (ty == 0);
    const bool sEdge = (ty == BY - 1) || (gy == ny - 1);

    /* =========================================================
     * PHASE 1: cur -> nxt, save interior in step1_z[]
     * ========================================================= */

    /* Register Z-pipeline for phase 1. */
    float vb1 = 0.0f, vc1 = 0.0f, vt1 = 0.0f;
    if (valid) {
        vc1 = __ldg(&cur[col]);
        vt1 = (NZ > 1) ? __ldg(&cur[nxny + col]) : vc1;
    }

    /* step1_z[z] holds the phase-1 output for Z-layer z (used by phase 2). */
    float step1_z[NZ];

    for (int z = 0; z < NZ; z++) {
        const int base = z * nxny;

        if (valid) {
            s[sy][sx] = vc1;
            if (wEdge) s[sy][0]      = __ldg(&cur[base + wcol]);
            if (eEdge) s[sy][BX + 1] = __ldg(&cur[base + ecol]);
            if (nEdge) s[0][sx]      = __ldg(&cur[base + ncol]);
            if (sEdge) s[SY - 1][sx] = __ldg(&cur[base + scol]);
        }
        __syncthreads();

        if (valid) {
            const float vn = s[sy - 1][sx];
            const float vs = s[sy + 1][sx];
            const float vw = s[sy][sx - 1];
            const float ve = s[sy][sx + 1];
            const float vb = (z == 0)      ? vc1 : vb1;
            const float vt = (z == NZ - 1) ? vc1 : vt1;
            const float pw = __ldg(&power[base + col]);

            const float result =
                vc1 * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt * d_ct + vb * d_cb +
                pw * d_dtCap + d_ambct;

            nxt[base + col] = result;
            step1_z[z] = result;   /* save for phase 2 */
        }
        __syncthreads();

        vb1 = vc1;
        vc1 = vt1;
        if (valid && z + 2 < NZ)
            vt1 = __ldg(&cur[(z + 2) * nxny + col]);
    }

    /* Grid-wide barrier: all blocks must finish phase 1 before any starts phase 2. */
    grid.sync();

    /* =========================================================
     * PHASE 2: nxt (halo) + step1_z[] (interior, registers) -> cur
     * ========================================================= */

    /* Prime phase-2 Z-pipeline from step1_z[] (registers, zero global reads). */
    float vb2 = 0.0f;
    float vc2 = valid ? step1_z[0] : 0.0f;
    float vt2 = (NZ > 1 && valid) ? step1_z[1] : vc2;

    for (int z = 0; z < NZ; z++) {
        const int base = z * nxny;

        if (valid) {
            /* Interior from register (NO global read for the centre column). */
            s[sy][sx] = vc2;
            /* Halo ring: read from nxt[] (= step-1 output) via read-only cache. */
            if (wEdge) s[sy][0]      = __ldg(&nxt[base + wcol]);
            if (eEdge) s[sy][BX + 1] = __ldg(&nxt[base + ecol]);
            if (nEdge) s[0][sx]      = __ldg(&nxt[base + ncol]);
            if (sEdge) s[SY - 1][sx] = __ldg(&nxt[base + scol]);
        }
        __syncthreads();

        if (valid) {
            const float vn = s[sy - 1][sx];
            const float vs = s[sy + 1][sx];
            const float vw = s[sy][sx - 1];
            const float ve = s[sy][sx + 1];
            const float vb = (z == 0)      ? vc2 : vb2;
            const float vt = (z == NZ - 1) ? vc2 : vt2;
            const float pw = __ldg(&power[base + col]);

            cur[base + col] =
                vc2 * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt * d_ct + vb * d_cb +
                pw * d_dtCap + d_ambct;
        }
        __syncthreads();

        vb2 = vc2;
        vc2 = vt2;
        /* Advance phase-2 pipeline from register array (no global reads). */
        if (valid && z + 2 < NZ)
            vt2 = step1_z[z + 2];
    }
}

/* ---- Fallback 1-step kernel (for odd niter or non-NZ=8 cases) ---- */
__global__ void stencil_1step(
        const float * __restrict__ cur,
        float       * __restrict__ nxt,
        const float * __restrict__ power,
        int nx, int ny, int nz)
{
    __shared__ float s[SY][SX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * BX + tx;
    const int gy = blockIdx.y * BY + ty;
    const int nxny = nx * ny;
    const bool valid = (gx < nx) && (gy < ny);

    const int col = gy * nx + gx;
    const int wcol = (gx > 0)      ? col - 1  : col;
    const int ecol = (gx < nx - 1) ? col + 1  : col;
    const int ncol = (gy > 0)      ? col - nx : col;
    const int scol = (gy < ny - 1) ? col + nx : col;

    const int sx = tx + 1;
    const int sy = ty + 1;

    const bool wEdge = (tx == 0);
    const bool eEdge = (tx == BX - 1) || (gx == nx - 1);
    const bool nEdge = (ty == 0);
    const bool sEdge = (ty == BY - 1) || (gy == ny - 1);

    float vb = 0.0f, vc = 0.0f, vt = 0.0f;
    if (valid) {
        vc = __ldg(&cur[col]);
        vt = (nz > 1) ? __ldg(&cur[nxny + col]) : vc;
    }

    for (int z = 0; z < nz; z++) {
        const int base = z * nxny;
        if (valid) {
            s[sy][sx] = vc;
            if (wEdge) s[sy][0]      = __ldg(&cur[base + wcol]);
            if (eEdge) s[sy][BX + 1] = __ldg(&cur[base + ecol]);
            if (nEdge) s[0][sx]      = __ldg(&cur[base + ncol]);
            if (sEdge) s[SY - 1][sx] = __ldg(&cur[base + scol]);
        }
        __syncthreads();
        if (valid) {
            const float vn = s[sy - 1][sx];
            const float vs = s[sy + 1][sx];
            const float vw = s[sy][sx - 1];
            const float ve = s[sy][sx + 1];
            const float vb_use = (z == 0)      ? vc : vb;
            const float vt_use = (z == nz - 1) ? vc : vt;
            const float pw = __ldg(&power[base + col]);
            nxt[base + col] =
                vc * d_cc +
                vn * d_cn + vs * d_cs +
                ve * d_ce + vw * d_cw +
                vt_use * d_ct + vb_use * d_cb +
                pw * d_dtCap + d_ambct;
        }
        __syncthreads();
        vb = vc; vc = vt;
        if (valid && z + 2 < nz)
            vt = __ldg(&cur[(z + 2) * nxny + col]);
    }
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <output_file>\n",
                argv[0]);
        return 1;
    }

    int nx    = atoi(argv[1]);
    int ny    = nx;
    int nz    = atoi(argv[2]);
    int niter = atoi(argv[3]);
    const char *outfile = argv[4];

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
    float ce = stepDivCap / Rx;
    float cw = ce;
    float cn = stepDivCap / Ry;
    float cs = cn;
    float ct = stepDivCap / Rz;
    float cb = ct;
    float cc = 1.0f - (2.0f*ce + 2.0f*cn + 3.0f*ct);
    float dtCap = dt / Cap;
    float ambct = ct * amb_temp;

    cudaMemcpyToSymbol(d_cc,    &cc,    sizeof(float));
    cudaMemcpyToSymbol(d_cn,    &cn,    sizeof(float));
    cudaMemcpyToSymbol(d_cs,    &cs,    sizeof(float));
    cudaMemcpyToSymbol(d_ce,    &ce,    sizeof(float));
    cudaMemcpyToSymbol(d_cw,    &cw,    sizeof(float));
    cudaMemcpyToSymbol(d_ct,    &ct,    sizeof(float));
    cudaMemcpyToSymbol(d_cb,    &cb,    sizeof(float));
    cudaMemcpyToSymbol(d_dtCap, &dtCap, sizeof(float));
    cudaMemcpyToSymbol(d_ambct, &ambct, sizeof(float));

    int size = nx * ny * nz;
    float *power = (float *)malloc(size * sizeof(float));
    float *buf0  = (float *)malloc(size * sizeof(float));
    if (!power || !buf0) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(7);
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                power[idx] = (rand() / (float)RAND_MAX) * 15.0f;
                buf0[idx]  = amb_temp + (rand() / (float)RAND_MAX) * 5.0f;
            }

    float *d_power, *d_a, *d_b;
    cudaMalloc(&d_power, size * sizeof(float));
    cudaMalloc(&d_a,     size * sizeof(float));
    cudaMalloc(&d_b,     size * sizeof(float));
    cudaMemcpy(d_power, power, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,     buf0,  size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BX, BY, 1);
    dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY, 1);

    /* Check if cooperative launch is supported and nz==8 for the 2-step path. */
    int coop_support = 0;
    cudaDeviceGetAttribute(&coop_support, cudaDevAttrCooperativeLaunch, 0);
    const bool use_2step = (coop_support != 0) && (nz == 8);

    /* d_cur: current t result (read input, written by phase 2).
     * d_nxt: step-1 intermediate (written by phase 1, read as halo in phase 2).
     * After each cooperative call, d_cur holds t+2; no pointer swap needed. */
    float *d_cur = d_a;
    float *d_nxt = d_b;

    cudaDeviceSynchronize();
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    if (use_2step) {
        /* Run floor(niter/2) cooperative 2-step calls. */
        int pairs = niter / 2;
        int remainder = niter % 2;

        void *args[] = { (void*)&d_cur, (void*)&d_nxt,
                         (void*)&d_power, (void*)&nx, (void*)&ny };

        for (int p = 0; p < pairs; p++) {
            cudaLaunchCooperativeKernel(
                (void*)stencil_2step<8>,
                grid, block, args);
        }

        /* Handle odd iteration (if niter is odd). */
        if (remainder) {
            /* After pairs*2 steps, d_cur holds the result.
             * Run one more 1-step: d_cur -> d_nxt, then swap. */
            stencil_1step<<<grid, block>>>(d_cur, d_nxt, d_power, nx, ny, nz);
            float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        }
    } else {
        /* Fallback: standard 1-step ping-pong. */
        for (int iter = 0; iter < niter; iter++) {
            stencil_1step<<<grid, block>>>(d_cur, d_nxt, d_power, nx, ny, nz);
            float *tmp = d_cur; d_cur = d_nxt; d_nxt = tmp;
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    fprintf(stderr, "compute_seconds: %.6f\n", elapsed);

    /* d_cur holds the final result. */
    cudaMemcpy(buf0, d_cur, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *fp = fopen(outfile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", outfile); return 1; }
    int index = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < nz; k++) {
                int idx = j + i*nx + k*nx*ny;
                fprintf(fp, "%d\t%g\n", index++, buf0[idx]);
            }
    fclose(fp);

    cudaFree(d_power); cudaFree(d_a); cudaFree(d_b);
    free(power); free(buf0);
    return 0;
}
