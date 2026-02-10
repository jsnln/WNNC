# Winding Number Normal Consistency (WNNC)

This repository contains the official implementation of the paper

**Fast and Globally Consistent Normal Orientation based on the Winding Number Normal Consistency**

**ACM ToG 2024 (SIGGRAPH Asia 2024 Journal Track)**

**[Siyou Lin](https://jsnln.github.io/), [Zuoqiang Shi](https://shizqi.github.io/), [Yebin Liu](http://liuyebin.com/)**

[Project page](https://jsnln.github.io/wnnc/index.html) | [Paper](https://arxiv.org/abs/2405.16634)

![WNNC](assets/wnnc_illustration.jpg "WNNC")

### Features

1. A PyTorch extension for accelerating winding numbers in `ext/wn_treecode`
2. A fast iterative method for normal estimation: `main_wnnc.py`
![WNNC gallery](assets/wnnc_gallery.jpg "WNNC gallery")
3. As a by-product, we provide *unofficial* CPU and CUDA implementations for [Gauss surface reconstruction](https://dl.acm.org/doi/10.1145/3233984) in `ext/gaussrecon_src`, which is faster than [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) with better smoothness control, but never officially open-sourced.

### Usage

1. For WNNC normal estimation:
```bash
cd ext
pip install -e .
cd ..

# width is important for normal quality, we provide a few presets through --width_config

# for clean uniform samples, use l0
python main_wnnc.py data/Armadillo_40000.xyz --width_config l0 --tqdm

# for noisy or non-uniform points, use configs l1 (small noise) ~ l5 (large noise) depending on the noise level
# a higher level gives smoother normals and better resilience to noise
python main_wnnc.py data/bunny_noised.xyz --width_config l1 --tqdm
...
python main_wnnc.py data/bunny_noised.xyz --width_config l5 --tqdm

# the user can also use custom widths:
python main_wnnc.py data/bunny_noised.xyz --width_config custom --wsmin 0.03 --wsmax 0.12 --tqdm

# to see a complete list of options:
python main_wnnc.py -h
```

2. For Gauss surface reconstruction:
First download [ANN 1.1.2](https://www.cs.umd.edu/~mount/ANN/) and unpack to `ext/gaussrecon_src/ANN`. Run `make` there. Then go back to the main repository directory, and:
```bash
sh build_GR_cpu.sh
sh build_GR_cuda.sh

./main_GaussRecon_cpu -i <input.xyz> -o <output.ply> -a <num_neighbors> -w <smoothing_width>
./main_GaussRecon_cuda -i <input.xyz> -o <output.ply> -a <num_neighbors> -w <smoothing_width>
```
Here, `-a` specifies the number of neighboring points used for estimating local areas. `-a 16` is usually an OK choice. `-a 0` would use a constant area value of 1E-5. `-w` specifies the smoothing width which should depend on the noise level of the point cloud.

**Note** This unofficial GR implementation does not use the *disk integration* technique and the *octree-based width selection* strategy in the original GR paper [Lu et al. 2018], so this is not a faithful reimplementation, but merely a by-product out of our winding number evaluation package.

### Related Research on Winding Numbers

[Surface Reconstruction from Point Clouds without Normals by Parametrizing the Gauss Formula  (ACM ToG 2022)](https://dl.acm.org/doi/10.1145/3554730)
Siyou Lin, Dong Xiao, Zuoqiang Shi, Bin Wang

[Surface Reconstruction Based on the Modified Gauss Formula  (ACM ToG 2018)](https://doi.org/10.1145/3233984)
Wenjia Lu, Zuoqiang Shi, Jian Sun,  Bin Wang

