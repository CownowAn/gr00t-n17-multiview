# gr00t-n17-multiview

A fork of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (GR00T N1.7, base commit `d2b7e75`)
with:

- `multiview/`: GR1 multi-view tabletop tasks (task env code, camera definitions, scripted demos),
  N1.7 fine-tuning and evaluation scripts, ColosseumV2 support. Start with `multiview/EVAL.md`.
- EE-VLA edge vision encoders for N1.7 (`gr00t/model/modules/edge_vision.py` and the model, processor
  and config changes it needs).

Files that upstream stores with Git LFS (`demo_data/`, `media/*.gif`, LFS files under `examples/` and
`scripts/deployment/`) are not included in this fork to keep it small. They are not used by `multiview/`;
get them from the upstream repository if you need them.

Licensed under Apache-2.0, as upstream (see `LICENSE`).
