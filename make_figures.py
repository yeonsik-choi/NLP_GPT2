"""
보고서 그림 생성 — figures/*.png

데이터는 각 실험 로그(learning_curve.log, multiseed.log, lora_hparam_sweep.log,
alpha_lr_sweep.log, multiepoch_experiment.log, sonnet_multiseed.log, degeneration.log)의
요약 수치를 그대로 옮긴 것이다(보고서 표와 동일). 라벨은 폰트 문제를 피해 영문 사용.

실행: python make_figures.py  →  figures/fig_*.png (150 dpi)
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})

STEPS = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
CURVES = {  # learning_curve.log (dev 2k, seed 11711)
  'full FT':       [.6465, .6480, .6060, .7450, .7305, .7595, .7810, .7820, .7685, .7910, .7995, .8020],
  'LoRA r=32':     [.6660, .6840, .7385, .6470, .7605, .7630, .7650, .7300, .7745, .7940, .8030, .8125],
  'LoRA a64/r=8':  [.5810, .6775, .7295, .7510, .7455, .7670, .7650, .7795, .7950, .7945, .7905, .7975],
}
COLORS = {'full FT': 'tab:red', 'LoRA r=32': 'tab:blue', 'LoRA a64/r=8': 'tab:green'}


def fig_learning_curve():
  fig, ax = plt.subplots(figsize=(5.5, 3.5))
  for name, ys in CURVES.items():
    ax.plot(STEPS, ys, marker='o', ms=3.5, label=name, color=COLORS[name])
  ax.set_xlabel('training step'); ax.set_ylabel('dev (2k subset) accuracy')
  ax.set_title('Learning curves, 3,000-step budget (seed 11711)')
  ax.legend(loc='lower right')
  fig.savefig('figures/fig_learning_curve.png'); plt.close(fig)


def fig_peft_efficiency():
  # multiseed.log + lora_hparam_sweep.log (3,000 step, 3-seed mean±std)
  rank = [  # (params%, mean, std, label)
    (0.118, .7647, .0047, 'r=4'), (0.235, .7769, .0011, 'r=8'),
    (0.470, .7768, .0091, 'r=16'), (0.935, .7887, .0145, 'r=32')]
  fig, ax = plt.subplots(figsize=(5.5, 3.5))
  xs, ys, es, _ = zip(*rank)
  ax.errorbar(xs, ys, yerr=es, marker='o', color='tab:blue', capsize=3, label='LoRA rank ladder (q/v, a=2r)')
  for x, y, _, lab in rank:
    ax.annotate(lab, (x, y), textcoords='offset points', xytext=(5, -11), fontsize=8)
  ax.errorbar([0.118], [.7717], yerr=[.0038], marker='s', color='tab:purple', capsize=3, label='ReFT r=8')
  ax.errorbar([0.235], [.7890], yerr=[.0051], marker='*', ms=13, color='tab:green',
              capsize=3, ls='none', label='LoRA a=64, r=8')
  ax.axhline(.8019, color='tab:red', ls='--', lw=1.2, label='full FT  0.8019 ± 0.0013')
  ax.fill_between([0.08, 130], .8019 - .0013, .8019 + .0013, color='tab:red', alpha=0.12)
  ax.set_xscale('log'); ax.set_xlim(0.08, 130)
  ax.set_xlabel('trainable parameters (% of 125M, log scale)')
  ax.set_ylabel('dev accuracy')
  ax.set_title('PEFT efficiency at equal budget (3,000 steps, 3 seeds)')
  ax.legend(loc='lower right', fontsize=8)
  fig.savefig('figures/fig_peft_efficiency.png'); plt.close(fig)


def fig_alpha_lr_heatmap():
  # alpha_lr_sweep.log (r=8, q/v, 3,000 step, 3-seed)
  mean = np.array([[.7501, .7769, .7863], [.7795, .7890, .7940]])
  std = np.array([[.0027, .0011, .0026], [.0022, .0051, .0075]])
  fig, ax = plt.subplots(figsize=(4.6, 2.9))
  im = ax.imshow(mean, cmap='YlGn', aspect='auto')
  for i in range(2):
    for j in range(3):
      ax.text(j, i, f'{mean[i, j]:.4f}\n±{std[i, j]:.4f}', ha='center', va='center', fontsize=9)
  ax.set_xticks(range(3), ['1e-4', '2e-4', '4e-4']); ax.set_yticks(range(2), ['a=16', 'a=64'])
  ax.set_xlabel('learning rate'); ax.set_ylabel('LoRA alpha (r=8)')
  ax.set_title('alpha x lr grid, dev acc (3 seeds)')
  ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.85)
  fig.savefig('figures/fig_alpha_lr_heatmap.png'); plt.close(fig)


def fig_convergence():
  # multiseed.log / lora_hparam_sweep.log (3,000 step) vs multiepoch_experiment.log (1 epoch)
  data = {  # name: ((mean, std)@3000, (mean, std)@1epoch)
    'full FT':      ((.8019, .0013), (.8600, .0077)),
    'LoRA r=32':    ((.7887, .0145), (.8430, .0015)),
    'LoRA a64/r=8': ((.7890, .0051), (.8379, .0053)),
  }
  fig, ax = plt.subplots(figsize=(5.5, 3.5))
  xs = [0, 1]
  for i, (name, (a, b)) in enumerate(data.items()):
    off = (i - 1) * 0.04
    ax.errorbar([x + off for x in xs], [a[0], b[0]], yerr=[a[1], b[1]],
                marker='o', capsize=4, label=name, color=COLORS[name])
  ax.set_xticks(xs, ['3,000 steps (~0.17 epoch, batch 8)', '17,688 steps (1 epoch, batch 16)'])
  ax.set_xlim(-0.35, 1.35)
  ax.set_ylabel('dev accuracy (3 seeds, mean ± std)')
  ax.set_title('Convergence: does PEFT catch up with full FT?')
  ax.legend(loc='lower right')
  fig.savefig('figures/fig_convergence.png'); plt.close(fig)


def fig_sonnet():
  # sonnet_multiseed.log (chrF) + degeneration.log (distinct-1, rep-4)
  strategies = ['temp1.2+top-p0.9', 'temp=1.0', 'top-p0.9 temp0.9', '+rep_penalty=1.2',
                'beam=3', 'beam=5', 'greedy']
  chrf = [42.06, 41.29, 41.12, 39.79, 32.19, 31.86, 31.70]
  err = [0.36, 0.34, 0.09, 1.30, 0, 0, 0]
  colors = ['tab:green'] * 4 + ['tab:gray'] * 3
  deg = {  # strategy -> (distinct-1, rep-4)
    'temp1.2+top-p0.9': (.754, .000), '+rep_penalty=1.2': (.998, .000),
    'beam=3': (.100, .861), 'greedy': (.123, .798)}

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={'width_ratios': [1.3, 1]})
  y = np.arange(len(strategies))[::-1]
  ax1.barh(y, chrf, xerr=err, color=colors, capsize=3, height=0.6)
  ax1.set_yticks(y, strategies, fontsize=9)
  ax1.set_xlabel('chrF (3 seeds, mean ± std)'); ax1.set_xlim(28, 44)
  ax1.set_title('Decoding strategy vs chrF')
  ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='tab:green'),
                      plt.Rectangle((0, 0), 1, 1, color='tab:gray')],
             labels=['sampling', 'deterministic'], loc='lower right', fontsize=8)

  names = list(deg)
  x = np.arange(len(names))
  ax2.bar(x - 0.18, [deg[n][0] for n in names], width=0.36, label='distinct-1', color='tab:blue')
  ax2.bar(x + 0.18, [deg[n][1] for n in names], width=0.36, label='rep-4', color='tab:orange')
  ax2.axhline(.754, color='tab:blue', ls=':', lw=1)
  ax2.set_xticks(x, ['best\nsampling', 'rep_pen\n1.2', 'beam=3', 'greedy'], fontsize=8)
  ax2.set_ylabel('ratio'); ax2.set_title('Degeneration metrics')
  ax2.legend(fontsize=8)
  fig.savefig('figures/fig_sonnet_chrf.png'); plt.close(fig)


if __name__ == '__main__':
  fig_learning_curve()
  fig_peft_efficiency()
  fig_alpha_lr_heatmap()
  fig_convergence()
  fig_sonnet()
  print('saved:', ', '.join(sorted(os.listdir('figures'))))
