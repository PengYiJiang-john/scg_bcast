"""Editable, vector overview of semantic counterfactual accountability.

All numerical values come from the explicit three-position deterministic
example below. This script performs no model calls and contains no empirical
LLM results. Run with Python 3, NumPy, and Matplotlib installed.
"""
from fractions import Fraction
from itertools import combinations
from math import factorial
from pathlib import Path
import argparse
from deterministic_example import loss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
W, H = 139.7, 190.0  # millimetres; ICLR text width
NAVY = '#203848'
TEAL = '#20797B'
RUST = '#B15432'
GRAY = '#596D77'
RULE = '#B7C7CE'
LIGHT = '#F1F5F6'
PALE_TEAL = '#EAF5F2'
PALE_RUST = '#FCEFE7'
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'text.color': NAVY,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',  # editable text; do not turn labels into paths
    'savefig.facecolor': 'white',
})

fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')


def text(x, y, label, size=7.5, color=NAVY, ha='left', va='center', bold=False):
    return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va=va,
                   fontweight='bold' if bold else 'normal', linespacing=1.1)


def box(x, y, w, h, fill='white', edge=RULE, lw=.7, radius=.75):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        linewidth=lw, facecolor=fill, edgecolor=edge)
    ax.add_patch(p)
    return p


def arrow(start, end, color=NAVY, lw=.8, head=7, curve=None):
    kw = {} if curve is None else {'connectionstyle': curve}
    p = FancyArrowPatch(start, end, arrowstyle='-|>', color=color,
                        linewidth=lw, mutation_scale=head,
                        shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(p)
    return p


def rule(y):
    ax.plot([4, W-4], [y, y], color=RULE, lw=.65)


def panel(y, label):
    text(4, y, label, size=9, bold=True, va='top')


def complete_state(x, y, w, heading, meaning, node=None, adverse=False,
                   decision=False):
    """A full message with an explicitly highlighted semantic node."""
    box(x, y, w, 26.5)
    text(x+w/2, y+2.8, heading, size=7.3, ha='center', bold=True)
    text(x+1.4, y+6.5, 'Records are', size=7.0)
    text(x+1.4, y+9.2, 'incomplete.', size=7.0)
    text(x+1.4, y+12.4, 'Source: audit log', size=7.0)
    color = RUST if adverse else TEAL
    fill = PALE_RUST if adverse else PALE_TEAL
    box(x+1.2, y+15.2, w-2.4, 10.0, fill=fill, edge=color, lw=.75, radius=.45)
    if node is not None:
        label = f'{node}: {meaning}'
    else:
        label = meaning
    text(x+w/2, y+20.2, label, size=7.0 if decision else 7.2, color=color, ha='center')


# A. Semantic nodes are embedded in factual, complete messages.
panel(3.3, 'A  Factual messages and semantic links')
text(4, 9.3,
     'Illustrative deterministic example; evidence and source remain separate from status.',
     size=7.1, color=GRAY)
xs = [4.0, 38.9, 73.8, 108.7]
ww = 27.0
complete_state(xs[0], 13.0, ww, 'Initial state', 'Compliance\nis not yet verified.', node='$s_0$')
complete_state(xs[1], 13.0, ww, 'After H', 'Compliance\nis confirmed.', node='$h$', adverse=True)
complete_state(xs[2], 13.0, ww, 'After R', 'Compliance\nis not yet verified.', node='$r$')
complete_state(xs[3], 13.0, ww, 'After B', 'Unverified.\nFurther review\nis required.', node='$y$', decision=True)
for left, right, label in zip(xs[:-1], xs[1:], ['H/A', 'R/B', 'B/C']):
    middle = (left+ww+right)/2
    text(middle, 23.1, label, size=7.0, ha='center', bold=True)
    arrow((left+ww+.4, 28.8), (right-.4, 28.8))
text(4, 44.4,
     'Owned links: H/A asserts certainty; R/B checks evidence; B/C selects review or approve.',
     size=7.1)
rule(48.2)


# B. This reference operator generates a draft, then patches only its target.
panel(51.3, 'B  Cancel R: patch its draft, then replay the decision')
text(4, 57.2,
     '$U=\\{H,B\\}$: keep H and B; replace R by its reference operation.',
     size=7.3)
bx = [4.0, 50.0, 100.0]
bw = [34.8, 34.8, 35.7]
for x, w, heading, status, adverse in zip(
        bx, bw,
        ['Current input', 'R-generated draft', 'Patched reference'],
        ['Compliance is\nconfirmed.', 'Compliance is\nnot yet verified.', 'Compliance is\nconfirmed.'],
        [True, False, True]):
    box(x, 62.2, w, 28.0)
    text(x+w/2, 65.2, heading, size=7.5, ha='center', bold=True)
    text(x+1.6, 70.2, 'Records are incomplete.', size=7.0)
    text(x+1.6, 74.6, 'Source: audit log', size=7.2)
    color = RUST if adverse else TEAL
    box(x+1.2, 78.3, w-2.4, 10.5,
        fill=PALE_RUST if adverse else PALE_TEAL, edge=color, radius=.45)
    text(x+w/2, 83.5, status, size=7.5, ha='center', color=color)
text(44.4, 70.2, 'Run R', size=7.1, ha='center')
arrow((39.4, 75.5), (49.4, 75.5))
text(92.4, 70.2, 'Patch', size=7.1, ha='center', color=RUST)
arrow((85.4, 75.5), (99.4, 75.5), color=RUST)
# Copy the target from the CURRENT input, never the historic pre-state.
ax.plot([21.4, 21.4, 117.85], [88.8, 93.5, 93.5], color=RUST, lw=.8)
arrow((117.85, 93.5), (117.85, 88.8), color=RUST)
text(4, 98.3, 'Copy the CURRENT input status; retain the other fields of the R draft.',
     size=7.4, color=RUST)
text(4, 104.5, 'Replay:', size=7.5, bold=True)
text(20.5, 104.5, 'H', size=8.0, bold=True)
arrow((25.5, 104.5), (31.5, 104.5))
text(34, 104.5, '$R^0$: confirmed', size=7.5, color=RUST)
arrow((64.2, 104.5), (70.2, 104.5))
text(72.7, 104.5, 'B re-runs', size=7.5, bold=True)
arrow((94.7, 104.5), (100.7, 104.5))
box(103, 101.8, 32.7, 5.4, fill=PALE_RUST, edge=RUST, radius=.45)
text(119.35, 104.5, 'APPROVE; loss = 1', size=7.4, color=RUST, ha='center')
text(4, 110.2,
     'For a native creation, its reference introduces no new target meaning.',
     size=7.1, color=GRAY)
rule(114.0)


# C. The replay invariant is a conditional mechanism, not an old output.
panel(117.0, 'C  Retain the mode; allow its realized behavior to change')
text(4, 122.0, 'R always checks the available evidence before stating verification status.',
     size=7.3)
box(67, 125.7, 31.0, 15.7, fill=LIGHT, edge=NAVY)
text(82.5, 133.55, 'Same R mode\nconditional rule', size=7.8,
     ha='center', bold=True)
for y, scenario, inp in [(125.7, 'Factual input', 'Confirmed'),
                          (134.4, 'H canceled', 'Unverified')]:
    text(4, y+3.5, scenario, size=7.4)
    box(29.5, y, 27.5, 7, fill=PALE_RUST if inp == 'Confirmed' else PALE_TEAL,
        edge=RUST if inp == 'Confirmed' else TEAL, radius=.6)
    text(43.25, y+3.5, inp, size=7.6, ha='center',
         color=RUST if inp == 'Confirmed' else TEAL)
    arrow((58.5, y+3.5), (65.5, y+3.5))
    arrow((99.5, y+3.5), (106.5, y+3.5), color=TEAL)
    box(108, y, 27.7, 7, fill=PALE_TEAL, edge=TEAL, radius=.6)
    text(121.85, y+3.5, 'Unverified', size=7.6, ha='center', color=TEAL)
text(4, 145.3, 'The same rule yields a correction above and a preservation below.',
     size=7.4, color=GRAY)
rule(149.5)


# D. Exact Shapley arithmetic from the deterministic example.
players = ('H', 'R', 'B')
universe = frozenset(players)
coalitions = [frozenset(c) for k in range(4) for c in combinations(players, k)]


def marginal(i, U):
    return loss(U | {i})-loss(U)


def weight(U):
    return Fraction(factorial(len(U))*factorial(2-len(U)), factorial(3))


psi = {i: sum((weight(U)*marginal(i, U) for U in coalitions if i not in U),
              Fraction(0)) for i in players}
assert tuple(psi.values()) == (Fraction(1, 6), Fraction(-1, 3), Fraction(1, 6))

panel(152.5, 'D  From replayed losses to Behavioral Shapley')
text(4, 158.0, '1 = retained; 0 = reference', size=7.1, color=GRAY)
text(56, 158.0, '$\\Delta_R(S)=v^L(S\\cup\\{R\\})-v^L(S)$', size=7.4)
left_cols = [8, 17, 26, 41]
for x, label in zip(left_cols, ['H', 'R', 'B', '$v^L(U)$']):
    text(x, 162.0, label, size=7.5, ha='center', bold=True)
ax.plot([4,49],[164.1,164.1], color=RULE, lw=.65)
rows = [frozenset(), frozenset('H'), frozenset('R'), frozenset('B'),
        frozenset(('H','R')), frozenset(('H','B')), frozenset(('R','B')), universe]
for j, U in enumerate(rows):
    y = 166.0+j*3.0
    if loss(U):
        ax.add_patch(Rectangle((4,y-1.4),45,2.8,facecolor=PALE_RUST,edgecolor='none'))
    for x, i in zip(left_cols[:3], players):
        text(x,y,str(int(i in U)),size=7.4,ha='center')
    text(left_cols[3],y,str(loss(U)),size=7.4,ha='center',
         color=RUST if loss(U) else NAVY,bold=bool(loss(U)))
ax.plot([4,49],[189.0,189.0], color=RULE,lw=.65)

right_cols = [61.0, 82.0, 105.0, 128.0]
for x, label in zip(right_cols, ['$S$', '$\\Delta_R(S)$', 'Weight', 'Product']):
    text(x,162.0,label,size=7.3,ha='center',bold=True)
ax.plot([55,135.7],[164.1,164.1], color=RULE,lw=.65)
for j, U in enumerate([frozenset(), frozenset('H'), frozenset('B'), frozenset(('H','B'))]):
    y = 167.0+j*4.8
    d,w = marginal('R',U),weight(U)
    lab = '$\\varnothing$' if not U else '$\\{'+','.join(i for i in players if i in U)+'\\}$'
    if d:
        ax.add_patch(Rectangle((55,y-2.2),80.7,4.4,facecolor=PALE_TEAL,edgecolor='none'))
    pair = f'{loss(U | {"R"})} − {loss(U)} = {str(d).replace("-", "−")}'
    for x, val in zip(right_cols, [lab,pair,str(w),str(w*d)]):
        text(x,y,val,size=7.4,ha='center',color=TEAL if d else NAVY)
ax.plot([55,135.7],[183.9,183.9],color=RULE,lw=.65)
text(55,187.0,'$\\psi_{(H,R,B)}=(1/6,\\;-1/3,\\;1/6)$',size=8.0,bold=True)


# Bounds checking catches clipped editable text before export.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
canvas = fig.bbox
outside = []
for artist in ax.texts:
    bb = artist.get_window_extent(renderer)
    if bb.x0 < canvas.x0-.5 or bb.x1 > canvas.x1+.5 or bb.y0 < canvas.y0-.5 or bb.y1 > canvas.y1+.5:
        outside.append(artist.get_text())
if outside:
    raise RuntimeError(f'Text falls outside the figure: {outside}')

base = OUT / 'scg_counterfactual_overview'
fig.savefig(base.with_suffix('.svg'))
fig.savefig(base.with_suffix('.pdf'))
fig.savefig(base.with_suffix('.png'), dpi=240)
plt.close(fig)
print('Saved editable SVG, vector PDF and PNG:', base)
print('Exact Behavioral Shapley:', tuple(str(psi[i]) for i in players))
