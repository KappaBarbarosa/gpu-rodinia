# SRAD (srad_v2) — Algorithm Specification

## Problem
Speckle Reducing Anisotropic Diffusion: iterative edge-preserving image
denoising. Operates on an `rows x cols` float image `J`, run for `niter`
iterations. The input image is generated **deterministically** from `srand(7)`
— there is NO external input file. You MUST reproduce the generation exactly.

## Deterministic input generation (reproduce EXACTLY)
1. `srand(7);`
2. For `i in 0..rows-1`, `j in 0..cols-1` (row-major): `I[k] = rand()/(float)RAND_MAX;`
3. `J[k] = (float)exp(I[k]);`  (this `J` is the image SRAD iterates on)

Neighbour-index arrays (clamped at borders):
- `iN[i]=i-1, iS[i]=i+1` then `iN[0]=0, iS[rows-1]=rows-1`
- `jW[j]=j-1, jE[j]=j+1` then `jW[0]=0, jE[cols-1]=cols-1`

## Per-iteration (repeat `niter` times)

### Step 1 — ROI statistics (small reduction)
Over the speckle box `i in [y1,y2], j in [x1,x2]` (let `size_R` = box area):
```
sum  = Σ J[k]
sum2 = Σ J[k]^2
meanROI = sum/size_R
varROI  = sum2/size_R - meanROI^2
q0sqr   = varROI / meanROI^2
```

### Step 2 — PREPARE pass (data-parallel over all pixels)
For each pixel `k = i*cols + j`, `Jc = J[k]`:
```
dN[k] = J[iN[i]*cols+j] - Jc
dS[k] = J[iS[i]*cols+j] - Jc
dW[k] = J[i*cols+jW[j]] - Jc
dE[k] = J[i*cols+jE[j]] - Jc
G2 = (dN^2+dS^2+dW^2+dE^2)/(Jc*Jc)
L  = (dN+dS+dW+dE)/Jc
num  = 0.5*G2 - (1.0/16.0)*(L*L)
den  = 1 + 0.25*L
qsqr = num/(den*den)
den  = (qsqr - q0sqr)/(q0sqr*(1+q0sqr))
c[k] = 1.0/(1.0+den)
clamp c[k] to [0,1]
```

### Step 3 — UPDATE pass (data-parallel over all pixels)
For each pixel `k`:
```
cN = c[k];  cW = c[k]
cS = c[iS[i]*cols+j]
cE = c[i*cols+jE[j]]
D  = cN*dN[k] + cS*dS[k] + cW*dW[k] + cE*dE[k]
J[k] = J[k] + 0.25*lambda*D
```

## Dependency / parallelism
- Step 1 is a reduction over the (small) ROI box → produces a scalar `q0sqr`.
- Steps 2 and 3 are **embarrassingly parallel over pixels** (one thread per
  pixel). BUT Step 3 reads `c[]` and `dS/dE` from neighbours, so Step 2 must
  fully finish (all `c[]`, `d*[]` written) before Step 3 starts — they cannot be
  fused into one pass without a barrier. Across iterations, `J` of iteration t+1
  depends on all of iteration t, so iterations are strictly sequential.
- A naive GPU port = 3 kernels per iteration (reduce, prepare, update), reading
  neighbours from global memory.

## Numerical type — CRITICAL
Arrays are `float`, but every per-pixel expression uses **double-precision
literals** (`0.5`, `1.0/16.0`, `.25`, `1.0`, `0.25*lambda`), so each `float`
store is the rounding of a `double` computation. To match the CPU golden
bit-closely (built with `--fmad=false`), compute the prepare/update math in
`double` and store to `float`, exactly as the serial reference.

## I/O contract (must match exactly)
Command line:
```
<prog> <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <niter> <output_file>
```
- No input files: generate `I`/`J` as above.
- `output_file`: the final `J` image, one line per pixel `"<index>\t<value>\n"`,
  row-major over `rows*cols` pixels, `index` from 0, value printed `%.5f`.
