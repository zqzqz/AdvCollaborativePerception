# AdvCollaborativePerception
Repo for USENIX security 2024 paper "On Data Fabrication in Collaborative Vehicular Perception: Attacks and Countermeasures" https://arxiv.org/abs/2309.12955

## Requirements
- Python package manager: conda
- GPU (tested on RTX 2080 Ti)
- Free space > 40 GB 

## Get started

```bash
git clone --recursive https://github.com/zqzqz/AdvCollaborativePerception.git
cd AdvCollaborativePerception
bash scripts/run.sh
```

The script will install dependencies, download dataset from Google Drives, set up the environment, and execute all evaluation tasks. Results are saved to `result` by default.

## Structure of the repository

- `data/OPV2V`: The perception dataset OPV2V and pre-computed meta files.
- `data/carla`: Pre-computed carla maps where OPV2V is collected.
- `data/model_3d`: 3D models that are useful for ray casting attacks.
- `models`: Pretrained models.
- `mvp`: The main module implementing our proposed attack and defense.
- `test`: Examples of using `mvp` to operate datasets, attack methods, etc.

## Debugging information

**Q:** Problems of CUDA and PyTorch.

**A:** Our script `scripts/run.sh` by default installs PyTorch 1.9.1 and CUDA 10.2. Please adjust the versions if it does not work for your environment.

**Q:** Deprecated functions of `numpy` or `shapely`.

**A:** Our code is tested on `numpy==1.19` and `shapely==1.8.1`. `numpy>=1.20` and `shapely==1.2` may throw warnings or errors about deprecated functions.