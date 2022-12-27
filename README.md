# Silk - An Open Source Cloth Simulation Library

TODO: add a banner here showing a few examples.

> This project is currently still in the early stages of development. It is not ready for end users yet, but if you wish to join the development, please get in touch!

## Introduction
The goal of this project is to simplify the development of cloth simulators.
We do this by bringing together powerful tools and modular components to piece together new simulators. 
Rather than providing a single, monolithic implementation of a specific simulator.

We want this project to contain a large set of educative examples of the implemented functionality.
You can find them in the [examples](examples) directory.

## Tools
This library relies on several great tools:

* [TinyAD](https://github.com/patr-schm/TinyAD): for automatic differentiation of energies on meshes. Derivatives of energies are crucial to many cloth simulators. By not having to derive and implement these by hand, the code is simpler and implementing new ideas is faster.
* [PolyScope](https://polyscope.run/): for flexible visualization of the simulated meshes, including data per mesh element (e.g. per vertex, edges etc.)
* [Eigen](https://eigen.tuxfamily.org/): for (sparse) matrices and linear algebra.
[libigl](https://libigl.github.io/)for geometry-related stuff, e.g. triangulation or getting the edges from a list of faces. 
* [IPC Toolkit](https://ipc-sim.github.io/ipc-toolkit/): for intersection-free collisions

Besides the above tools, I am also thankful for these resources:
* [Dynamic Deformables course](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/): for the great explanation of the deformation gradient and potential energies.
* Numerical Optimization, the book by Jorge Nocedal and Stephen J. Wright, in particular for its great explanation of line search methods. 

## Installation and building
Currently, I'm only using CMake for dependency management and building. 
Dependencies should be downloaded automatically. 
I use the CMake VSCode integration for configuring and building. 
However, if you want to build from the command line, you can use these commands:
```bash
cd ~/silk
mkdir build
cd build
cmake ..
make -j 12
```
Then you can run any of the built executables, e.g.
```
./falling_towel
```

## Feature wishlist
* Procedural and parametric meshes of clothes, we can take inspiration from [Sensitive Couture](https://www.cs.columbia.edu/cg/SC/) and the [Berkeley Garment Library](http://graphics.berkeley.edu/resources/GarmentLibrary/).
* Seams, as presented in the [True Seams](https://gabrielcirio.gitlab.io/projects/trueseams/trueseams.html) paper.
* Python bindings