"""Generate Figure 1 (speedup bar chart) and Figure 2 (Phase-2 trajectory)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Academic paper style
# ---------------------------------------------------------------------------
OPUS_C   = '#2166ac'
SONNET_C = '#d6604d'
HAIKU_C  = '#4dac26'

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'legend.framealpha':  0.9,
    'legend.edgecolor':  '#aaaaaa',
    'axes.linewidth':     0.7,
    'xtick.major.width':  0.7,
    'ytick.major.width':  0.7,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'figure.dpi':         150,
})

# ===========================================================================
# Figure 1 — Speedup grouped bar chart (log scale)
# ===========================================================================
serial_ms = {
    'hotspot':    5850,
    'hotspot3D':   711,
    'nw':            9,
    'bfs':          77,
    'srad':       6650,
    'lavaMD':     2020,
    'pathfinder':  133,
    'kmeans':      171,
}
best_ms = {               # (opus_ms, sonnet_ms, haiku_ms)  None = never correct
    'hotspot':    (33.7,  33.5,  None),
    'hotspot3D':  ( 6.56,  6.66,  6.49),
    'nw':         ( 1.46, 11.0,  None),
    'bfs':        ( 0.79,  0.82,  1.23),
    'srad':       (16.1,  62.0,  16.1),
    'lavaMD':     ( 0.70,  1.29,  0.75),
    'pathfinder': ( 1.04,  1.10,  1.21),
    'kmeans':     ( 0.50,  0.40,  0.53),
}
benches = list(best_ms.keys())
xlabels = ['hotspot', 'hotspot3D', 'NW', 'BFS', 'SRAD', 'lavaMD', 'pathfinder', 'kmeans']

def sp(bench, ms):
    return None if ms is None else serial_ms[bench] / ms

fig1, ax1 = plt.subplots(figsize=(7.0, 3.0))

x = np.arange(len(benches))
w = 0.25
for idx, (offset, color, mname) in enumerate(zip([-w, 0, w],
                                                   [OPUS_C, SONNET_C, HAIKU_C],
                                                   ['Opus', 'Sonnet', 'Haiku'])):
    vals = [sp(b, best_ms[b][idx]) or 0 for b in benches]
    ax1.bar(x + offset, vals, width=w, color=color, label=mname,
            alpha=0.85, zorder=3, linewidth=0.3, edgecolor='white')
    for xi, bench in enumerate(benches):
        if best_ms[bench][idx] is None:
            ax1.text(xi + offset, 0.62, 'x', ha='center', va='bottom',
                     fontsize=7.5, color=color, fontweight='bold')

ax1.set_yscale('log')
ax1.set_ylabel('Speedup over serial (×)')
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, rotation=15, ha='right')
ax1.set_ylim(bottom=0.5)
ax1.yaxis.grid(True, which='major', color='#cccccc', linewidth=0.5, zorder=0)
ax1.yaxis.grid(True, which='minor', color='#eeeeee', linewidth=0.3, zorder=0)
ax1.xaxis.grid(False)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='upper left', ncol=3)
ax1.set_title('Best correct GPU kernel speedup per model  (x = never produced correct output)')

fig1.tight_layout()
fig1.savefig('fig1_speedup.pdf', bbox_inches='tight')
fig1.savefig('fig1_speedup.png', bbox_inches='tight')
print('Saved fig1_speedup.pdf')

# ===========================================================================
# Figure 2 — Phase-2 optimization trajectories (lavaMD + SRAD)
# ===========================================================================
# Encoding per stage: (label, kernel_ms_or_None, ok)
#   ok=True  → correct result, kernel_ms is real
#   ok=False → incorrect/build-failed:
#              km is not None  → output produced but wrong (show × at top)
#              km is None      → no output at all (show × at top)
#   ok=None  → padding (ignore)

TRAJ = {
    'lavaMD': {
        (OPUS_C,   'Opus'):   [('P1', 1.300, True), ('R1', 0.792, True),
                                ('R2', 0.721, True), ('R3', 0.697, True)],
        (SONNET_C, 'Sonnet'): [('P1', 1.293, True), ('R1', 0.842, False),
                                ('R2', None,  False),('R3', None,  None)],
        (HAIKU_C,  'Haiku'):  [('P1', 1.300, True), ('R1', 1.017, False),
                                ('R2', 0.895, True), ('R3', 0.753, True)],
    },
    'srad': {
        (OPUS_C,   'Opus'):   [('P1', None,  False), ('R1', 16.06, True),
                                ('R2', 41.27, True),  ('R3', 19.70, True)],
        (SONNET_C, 'Sonnet'): [('P1', None,  False), ('R1', None,  False),
                                ('R2', 94.08, True),  ('R3', 62.00, True)],
        (HAIKU_C,  'Haiku'):  [('P1', 240.7, True),  ('R1', 18.09, True),
                                ('R2', 16.54, True),  ('R3', 16.08, True)],
    },
}

def plot_trajectory(ax, title, trajs, logy=False, ylabel=True):
    """
    Solid line + filled circle = best-so-far over correct stages.
    Coloured × at the TOP of axes = any failed stage (ok=False), regardless
    of whether a kernel time exists. Three models get slight x-offsets so
    overlapping ×'s are visible.
    """
    ax.set_title(title, fontsize=9, fontweight='bold')
    if ylabel:
        ax.set_ylabel('Kernel time (ms)')
    ax.set_xlabel('Optimization stage')
    if logy:
        ax.set_yscale('log')

    all_labels = [t[0] for t in list(trajs.values())[0]]
    model_list = list(trajs.keys())           # ordered: Opus, Sonnet, Haiku
    n_models   = len(model_list)
    x_offsets  = np.linspace(-0.15, 0.15, n_models)   # slight horizontal spread

    # --- draw best-so-far lines first ---
    for (color, mname), traj in trajs.items():
        best_x, best_y, best = [], [], None
        for xi, (_, km, ok) in enumerate(traj):
            if ok is True and km is not None:
                best = min(best, km) if best is not None else km
                best_x.append(xi)
                best_y.append(best)
        ax.plot(best_x, best_y, '-o', color=color, label=mname,
                linewidth=1.8, markersize=5.5, zorder=4)

    # --- draw × at top for failed stages ---
    fail_legend_added = False
    for mi, ((color, mname), traj) in enumerate(trajs.items()):
        xoff = x_offsets[mi]
        for xi, (_, km, ok) in enumerate(traj):
            if ok is False:   # explicit failure; ok=None is padding, skip
                ax.plot(xi + xoff, 0.97, 'x',
                        transform=ax.get_xaxis_transform(),
                        color=color, markersize=8, markeredgewidth=2.0,
                        clip_on=False, zorder=6, label='_nolegend_')
    # Single neutral legend entry for the × symbol
    ax.plot([], [], 'x', color='#555555', markersize=7, markeredgewidth=1.8,
            label='No correct output')

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.yaxis.grid(True, which='major', color='#cccccc', linewidth=0.5, zorder=0)
    ax.yaxis.grid(True, which='minor', color='#eeeeee', linewidth=0.3, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # No per-axes legend — caller collects handles for a shared figure legend

fig2, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))

plot_trajectory(axes[0], '(a) lavaMD', TRAJ['lavaMD'], logy=False, ylabel=True)
plot_trajectory(axes[1], '(b) SRAD',   TRAJ['srad'],   logy=True,  ylabel=False)

# Shared figure-level legend below both panels
handles, lbls = axes[0].get_legend_handles_labels()
seen, uh, ul = set(), [], []
for h, l in zip(handles, lbls):
    if l not in seen:
        seen.add(l); uh.append(h); ul.append(l)
fig2.legend(uh, ul, loc='lower center', bbox_to_anchor=(0.5, 0.0),
            ncol=4, fontsize=8, framealpha=0.9, edgecolor='#aaaaaa')

fig2.suptitle(
    'Phase-2 optimization trajectories  '
    '(line = best-so-far correct kernel time; '
    'x = no correct output at that stage)',
    fontsize=8)
fig2.tight_layout(rect=[0, 0.12, 1, 0.97])
fig2.savefig('fig2_trajectory.pdf', bbox_inches='tight')
fig2.savefig('fig2_trajectory.png', bbox_inches='tight')
print('Saved fig2_trajectory.pdf')
