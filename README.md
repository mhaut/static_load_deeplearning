# Training Deep Networks: A Static Load Balancing Approach
The Code for "Training Deep Networks: A Static Load Balancing Approach". [https://link.springer.com/article/10.1007%2Fs11227-020-03200-6]
```
S. Moreno-Alvarez, J. M. Haut, M. E. Paoletti, J. A. Rico-Gallego, J. C. Diaz-Martin and J. Plaza.
Training Deep Networks: A Static Load Balancing Approach.
Journal of Supercomputing.
DOI: 10.1007/s11227-020-03200-6
Accepted for publication, 2020.
```

![static](https://github.com/mhaut/static_load_deeplearning/blob/master/images/architecture.png)


## Requirements:

The code is built with following libraries:

- [FuPerMod](https://www.researchgate.net/publication/266390431_fupermod-120tar). Apply our patch:
```bash
cd patch_fupermod
sh patch_fupermod.sh
```
- [CUDA 9.x or higher](https://developer.nvidia.com/cuda-downloads)
- [MPI 3.x or higher](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/) 1.0.1


**LAUNCH**
Complete the FILL variables in [launch_experiment.sh](https://github.com/mhaut/static_load_deeplearning/blob/master/launch_experiment.sh).

```bash
sh launch_experiment.sh
```
