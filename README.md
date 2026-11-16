# ai-excited-01 — Molten Ribbon

A procedural space scene: a colossal molten plasma ribbon, drawn glowing
pink/magenta into orbit around a small dark world with a blue rim, set against
a starry wine-purple sky.

## Contents

- `gen.py` — procedural generator (numpy + Pillow, fully seeded). Renders the
  scene at 2048x2048 and downsamples to the final 1024x1024 `entry.png`.
  It builds the ribbon from a Catmull-Rom spline with a spiral siphon around
  the planet, adds flow-aligned turbulence, filaments, shed embers, a wide
  corona glow, and a simple tonemap grade.
- `main.ts` — entry driver (Node 22, stdlib only). Copies `entry.png` from the
  repo root into `${MAKEFUN_OUTPUT_DIR:-/output}/entry.png`.

## Reproduce the image

```
python3 gen.py        # writes variant_0/1/2.png
```

Variant 2 (seed 37) is the shipped `entry.png`.
