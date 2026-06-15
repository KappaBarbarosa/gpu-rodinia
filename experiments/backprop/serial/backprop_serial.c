/*
 * Backpropagation (backprop) — serial reference implementation.
 *
 * Port of the Rodinia backprop benchmark (openmp/backprop/*.c) made
 * self-contained and instrumented for the experiment harness. A single
 * three-layer fully-connected neural network (n_input inputs, 16 hidden, 1
 * output, each layer carrying an extra index-0 "threshold/bias" unit) is built
 * with deterministically generated input units and initial weights from
 * srand(7), then trained for EXACTLY ONE epoch (one forward pass + one
 * backprop pass). There is NO external input file — everything is regenerated
 * from the seed, so a blind author can reproduce identical state.
 *
 * The math is copied verbatim from openmp/backprop/backprop.c:
 *   squash(x) = 1/(1+exp(-x))                          (sigmoid)
 *   forward:   l2[j] = squash( sum_k conn[k][j]*l1[k] ), k=0..n1 (k=0 = bias)
 *   out delta: delta_o[j] = o*(1-o)*(t-o)              t = target = 0.1
 *   hid delta: delta_h[j] = h*(1-h)* sum_k delta_o[k]*who[j][k]
 *   adjust:    new_dw = ETA*delta[j]*ly[k] + MOMENTUM*oldw[k][j]
 *              w[k][j]   += new_dw;  oldw[k][j] = new_dw
 * with ETA=0.3, MOMENTUM=0.3 (from backprop.h). All initial prev-weights are 0.
 *
 * PARALLELISM: the dominant work is the input->hidden forward weighted sum and
 * the input->hidden weight adjustment, both data-parallel over the
 * (n_input+1) x (hidden+1) weight matrix. The forward sum over the n_input
 * input units is a reduction whose order a GPU port will REORDER; backprop is
 * smooth (sigmoid + weighted sums) so this is numerically benign.
 *
 * OUTPUT (the oracle): the adjusted input->hidden weight matrix
 * input_weights[i][j], i=0..n_input, j=0..hidden, row-major, one value/line
 * "<index>\t<value>\n", %.6f. Size = (n_input+1)*(hidden+1).
 *
 * CLI:  backprop_serial <n_input> <output_file>
 *
 * Build: gcc -O2 -o backprop_serial backprop_serial.c -lm
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define HIDDEN   16
#define OUTPUT   1
#define ETA      0.3f   /* learning rate     (backprop.h: ETA 0.3)      */
#define MOMENTUM 0.3f   /* momentum          (backprop.h: MOMENTUM 0.3) */

static double now_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* sigmoid squashing function, verbatim from backprop.c */
static float squash(float x) {
    return (1.0 / (1.0 + exp(-x)));
}

static float *alloc1(int n)        { return (float *)malloc((unsigned)(n * sizeof(float))); }
static float **alloc2(int m, int n) {
    float **w = (float **)malloc((unsigned)(m * sizeof(float *)));
    for (int i = 0; i < m; i++) w[i] = alloc1(n);
    return w;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <n_input> <output_file>\n", argv[0]);
        return 1;
    }
    int n_in  = atoi(argv[1]);     /* number of input units */
    const char *ofile = argv[2];
    int hid = HIDDEN;
    int out = OUTPUT;

    /* ---- allocate network (sizes +1 for the index-0 bias/threshold unit) ---- */
    float  *input_units  = alloc1(n_in + 1);
    float  *hidden_units = alloc1(hid  + 1);
    float  *output_units = alloc1(out  + 1);
    float  *hidden_delta = alloc1(hid  + 1);
    float  *output_delta = alloc1(out  + 1);
    float  *target       = alloc1(out  + 1);
    float **input_weights       = alloc2(n_in + 1, hid + 1);
    float **hidden_weights      = alloc2(hid  + 1, out + 1);
    float **input_prev_weights  = alloc2(n_in + 1, hid + 1);
    float **hidden_prev_weights = alloc2(hid  + 1, out + 1);

    /* ============================================================
     * Deterministic state generation — PINNED seed=7.
     * RNG: C library rand(); weights/inputs use rand()/RAND_MAX (NOT drnd).
     * Call order EXACTLY matches Rodinia (setup -> bpnn_initialize(7) ->
     * bpnn_create -> load):
     *   1. bpnn_randomize_weights(input_weights,  n_in, hid):
     *        i=0..n_in, j=0..hid : (n_in+1)*(hid+1) rand() calls
     *   2. bpnn_randomize_weights(hidden_weights, hid,  out):
     *        i=0..hid,  j=0..out : (hid+1)*(out+1) rand() calls
     *   3. bpnn_zero_weights(input_prev_weights)   -> all 0, no rand
     *   4. bpnn_zero_weights(hidden_prev_weights)  -> all 0, no rand
     *   5. bpnn_randomize_row(target, out): target[i]=0.1, i=0..out, no rand
     *   6. load(): input_units[k]=rand()/RAND_MAX, k=1..n_in : n_in rand() calls
     *      (input_units[0] is set to 1.0 by the forward pass, not here)
     * ============================================================ */
    srand(7);

    for (int i = 0; i <= n_in; i++)
        for (int j = 0; j <= hid; j++)
            input_weights[i][j] = (float) rand() / RAND_MAX;

    for (int i = 0; i <= hid; i++)
        for (int j = 0; j <= out; j++)
            hidden_weights[i][j] = (float) rand() / RAND_MAX;

    for (int i = 0; i <= n_in; i++)
        for (int j = 0; j <= hid; j++)
            input_prev_weights[i][j] = 0.0;

    for (int i = 0; i <= hid; i++)
        for (int j = 0; j <= out; j++)
            hidden_prev_weights[i][j] = 0.0;

    for (int i = 0; i <= out; i++)
        target[i] = 0.1;

    for (int k = 1; k <= n_in; k++)
        input_units[k] = (float) rand() / RAND_MAX;

    /* ======================== TRAIN KERNEL (timed) ======================== */
    double t0 = now_seconds();

    /* --- forward: input -> hidden --- */
    input_units[0] = 1.0;                       /* bias unit */
    for (int j = 1; j <= hid; j++) {
        float sum = 0.0;
        for (int k = 0; k <= n_in; k++)
            sum += input_weights[k][j] * input_units[k];
        hidden_units[j] = squash(sum);
    }

    /* --- forward: hidden -> output --- */
    hidden_units[0] = 1.0;                       /* bias unit */
    for (int j = 1; j <= out; j++) {
        float sum = 0.0;
        for (int k = 0; k <= hid; k++)
            sum += hidden_weights[k][j] * hidden_units[k];
        output_units[j] = squash(sum);
    }

    /* --- output error / delta --- */
    for (int j = 1; j <= out; j++) {
        float o = output_units[j];
        float t = target[j];
        output_delta[j] = o * (1.0 - o) * (t - o);
    }

    /* --- hidden error / delta --- */
    for (int j = 1; j <= hid; j++) {
        float h = hidden_units[j];
        float sum = 0.0;
        for (int k = 1; k <= out; k++)
            sum += output_delta[k] * hidden_weights[j][k];
        hidden_delta[j] = h * (1.0 - h) * sum;
    }

    /* --- adjust hidden -> output weights --- */
    hidden_units[0] = 1.0;
    for (int j = 1; j <= out; j++) {
        for (int k = 0; k <= hid; k++) {
            float new_dw = (ETA * output_delta[j] * hidden_units[k])
                         + (MOMENTUM * hidden_prev_weights[k][j]);
            hidden_weights[k][j]      += new_dw;
            hidden_prev_weights[k][j]  = new_dw;
        }
    }

    /* --- adjust input -> hidden weights (the data-parallel hot loop) --- */
    input_units[0] = 1.0;
    for (int j = 1; j <= hid; j++) {
        for (int k = 0; k <= n_in; k++) {
            float new_dw = (ETA * hidden_delta[j] * input_units[k])
                         + (MOMENTUM * input_prev_weights[k][j]);
            input_weights[k][j]      += new_dw;
            input_prev_weights[k][j]  = new_dw;
        }
    }

    double t1 = now_seconds();
    fprintf(stderr, "compute_seconds: %.6f\n", t1 - t0);

    /* ---- output: adjusted input->hidden weight matrix, row-major ---- */
    FILE *fp = fopen(ofile, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", ofile); return 1; }
    int idx = 0;
    for (int i = 0; i <= n_in; i++)
        for (int j = 0; j <= hid; j++)
            fprintf(fp, "%d\t%.6f\n", idx++, input_weights[i][j]);
    fclose(fp);

    return 0;
}
