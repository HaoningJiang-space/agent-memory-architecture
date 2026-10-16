"""Standalone figures for paper-anchored constraints and bounded block replay."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]/'evidence/architecture'
    resources = json.loads((root/'supply_resource_bounds.json').read_text())
    data = json.loads(gzip.decompress((root/'bank_supply.json.gz').read_bytes()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    ax = axes[0]
    points = resources['reference']
    x = np.linspace(0, 54, 100)
    for m, color in [(1,'#7570b3'), (4,'#d95f02'), (8,'#1b9e77')]:
        ax.plot(x, x*m/.8, '--', color=color, alpha=.75, label=f'Required, M={m}, r=0.8')
    ax.plot([p['hb_GB_s'] for p in points], [p['peak_GFLOPs'] for p in points],
            'o-', color='#222222', label='H2-LLM fitting MAC count')
    for p in points:
        ax.annotate(f"{p['macs']} MAC", (p['hb_GB_s'], p['peak_GFLOPs']),
                    xytext=(3, 7), textcoords='offset points', fontsize=8)
    ax.set(xlabel='HB interface per PE (GB/s)', ylabel='Compute per PE (GFLOP/s)',
           ylim=(0,310), title='40 nm, 1 GHz paper reference\nDecoder/network area still excluded')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=.2)
    ax = axes[1]
    labels = ['ideal_pool','routed_g4_x1','routed_g16_x1','routed_g16_x2']
    names = ['Zero-fabric\n16-bank pool','4-bank pool\n1x links','16-bank pool\n1x links','16-bank pool\n2x links']
    for i, label in enumerate(labels):
        values = np.array([r['variants']['local']['cycles']/r['variants'][label]['cycles']
                           for r in data['results']])
        # Fixed jitter is only visual; every one of the 704 conditions shown.
        jitter = .14*np.sin(np.arange(len(values))*2.39996)
        ax.scatter(i+jitter, values, s=5, alpha=.17, color='#2468a0')
        ax.plot([i-.2,i+.2], [np.median(values)]*2, color='#b54423', linewidth=2)
    ax.axhline(1, color='black', linestyle='--', linewidth=.8)
    ax.set(xticks=range(len(labels)), xticklabels=names, ylabel='Local cycles / pool cycles',
           title='Finite block supply, 704 controlled cases\nNo area, energy or end-to-end speed claim', ylim=(.45,1.34))
    ax.grid(axis='y', alpha=.2)
    fig.savefig(root/'bank_supply.png', dpi=180)
    fig.savefig(root/'bank_supply.pdf')


if __name__ == '__main__':
    main()
