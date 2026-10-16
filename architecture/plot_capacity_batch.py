"""Plot measured diagnostic results without presenting failed controls as gains."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'evidence/architecture'


def main():
    data = json.loads((ROOT / 'capacity_batch_analysis.json').read_text())
    rows = data['comparisons']
    assert all(not r['numerical_pass'] for r in rows)
    labels = [f"{r['strategy'].replace('raw_512', 'small raw')}\nB={r['batch']}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout='constrained')
    for ax, key, title in zip(axes,
            ('decode_throughput_ratio', 'per_request_step_ratio'),
            ('Aggregate forward throughput / serial', 'Per-request step time / matched singles')):
        bars = ax.bar(range(len(rows)), [r[key] for r in rows], color='#bac5d4',
                      edgecolor='#4b5b70', hatch='//')
        ax.axhline(1, color='black', linewidth=0.8)
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Ratio')
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
        ax.set_ylim(0, max(r[key] for r in rows)*1.16)
    fig.suptitle('A800 / Qwen3-32B BF16: 16 teacher-forced steps; identical request sets\n'
                 'Hatched timings: all batched mappings exceeded the fixed logit tolerance', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / f'capacity_batch_timing.{ext}', dpi=160)
    plt.close(fig)

    data = json.loads((ROOT / 'attention_mask_probe.json').read_text())
    assert len(data['cases']) == 9 and all(r['numerical_pass'] for r in data['cases'])
    strategies = ('raw', 'raw_512', 'mem0')
    routes = ('native_no_padding', 'expanded_no_padding', 'hf_padded')
    titles = ('Native GQA', 'Explicit KV expansion', 'HF padded/masked path')
    colors = ('#176b87', '#d9912e', '#9e456b')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout='constrained')
    for j, (route, title, color) in enumerate(zip(routes, titles, colors)):
        selected = [next(r for r in data['cases'] if r['strategy'] == s and r['route'] == route)
                    for s in strategies]
        x = np.arange(3)+(j-1)*0.25
        axes[0].bar(x, [statistics.mean(r['forward_ms']) for r in selected], width=.24,
                    label=title, color=color)
        axes[1].bar(x, [r['peak_extra_bytes']/2**20 for r in selected], width=.24,
                    label=title, color=color)
    for ax in axes:
        ax.set_xticks(range(3), ['21,032 tokens', '2,669 tokens', '255 tokens'])
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean forward time (ms, log scale)')
    axes[1].set_ylabel('Peak allocation above inputs (MiB, log scale)')
    fig.suptitle('A800 single-layer control: 64 Q heads / 8 KV heads, BF16 synthetic tensors\n'
                 'Same valid attention; all nine cases passed independent FP32 reference checks', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT / f'attention_mask_control.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
