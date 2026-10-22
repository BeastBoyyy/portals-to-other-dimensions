# Ablation (synthetic suite, 3 seeds x 15 s per clip; see REPORT.md for caveats)

| Variant | jitter px | acc% still | acc% fast | acc% tilt | acc% mean | outlier% | swaps | render err px slow / fast (median) | pops | false-hand % |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (all on) | 0.25 | 99.48 | 95.35 | 91.92 | 97.01 | 0.805 | 0 | 6.2 / 85.5 | 119 | 0.000 |
| no gates (4.3 off) | 11.04 | 99.70 | 96.38 | 92.62 | 97.72 | 0.855 | 0 | 6.2 / 85.5 | 487 | 100.000 |
| no temporal gate | 0.25 | 99.93 | 95.80 | 92.38 | 97.68 | 0.849 | 0 | 6.2 / 85.1 | 115 | 2.153 |
| no motion gate | 1.91 | 99.41 | 95.97 | 92.03 | 97.16 | 0.691 | 0 | 6.2 / 85.4 | 244 | 0.000 |
| no geometry gate | 0.25 | 99.48 | 95.35 | 91.92 | 97.01 | 0.805 | 0 | 6.2 / 85.5 | 119 | 0.000 |
| no identity tracking (x-order) | 48.61 | 99.44 | 95.82 | 91.28 | 91.83 | 6.624 | 43 | 6.3 / 86.4 | 1759 | 0.000 |
| kinematic constraint ON | 0.40 | 99.48 | 95.20 | 91.92 | 96.75 | 0.968 | 0 | 6.2 / 85.4 | 124 | 0.000 |
| no anchor-spike gate | 2.28 | 98.66 | 95.22 | 94.28 | 95.97 | 1.935 | 0 | 6.2 / 85.4 | 455 | 0.000 |
| filter: none | 0.76 | 99.46 | 96.95 | 94.16 | 97.28 | 0.832 | 0 | 4.2 / 81.6 | 147 | 0.000 |
| filter: moving average (3) | 0.46 | 99.44 | 1.38 | 4.96 | 58.89 | 16.766 | 0 | 9.6 / 115.3 | 132 | 0.000 |
| prediction off | 0.14 | 99.48 | 95.35 | 91.92 | 97.01 | 0.805 | 0 | 18.1 / 119.4 | 71 | 0.000 |
| person-mask gate OFF (poster x2) | - | - | - | - | - | - | - | - / - | - | 99.184 |
| person-mask gate ON  (poster x2) | - | - | - | - | - | - | - | - / - | - | 0.000 |
