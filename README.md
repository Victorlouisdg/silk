# Silk - An Experimental Open Source Cloth Simulation Library

The goal of this project is to simplify the development of cloth simulations.

As part of this goal we hope to provide implementations of several notable cloth simulators:
* A Mass-Spring-Damper Simulation
* A Baraff-Witkin-based simulator
* Codimensional Incremental Potential Contact (C-IPC) 

We wish to provide easy access to:
* Linear algebra and solvers
* Commonly used energies and forces
* Meshes to simulate
* Collision detection algorithms and other geometry queries
* Parallellism and GPU-acceleration
* Automatic differentation
* A test suite, benchmarking and profiling
* Visualization and video generation
* .obj import and export
* vmap? (automatic vectorization)

In this project we highly value:
* Readability
* Educational value
* Extensibility
* Robustness


## Technologies
Currently the plan is to make this a C++/python project.

Initially I will try combining the PyTorch C++ API (libtorch) with Eigen to write an initial prototype.
Other interesting libraries to keep in mind are:
* autodiff
* TinyAD
* JAX
* tinygrad
* open3d
* libigl
* PolyFEM
* geometry-central
* polyscope

## Installation
Place into libs:
* https://pytorch.org/cppdocs/installing.html
* https://polyscope.run/building/
* http://geometry-central.net/build/building/

## Building
```
cmake -DCMAKE_PREFIX_PATH=/home/idlab185/silk/libs/libtorch ..

make -j 12
# or
cmake --build . --config Release
```