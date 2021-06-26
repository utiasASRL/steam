# steam

STEAM (Simultaneous Trajectory Estimation and Mapping) Engine is an optimization library aimed at solving batch nonlinear optimization problems involving both SO(3)/SE(3) and continuous-time components. This is accomplished by using an iterative Gauss-Newton-style estimator in combination with techniques developed and used by ASRL. With respect to SO(3) and SE(3) components, we make use of the constraint sensitive perturbation schemes discussed in Barfoot and Furgale [1]. STEAM Engine is by no means intended to be the fastest car on the track; the intent is simply to be fast enough for the types of problems we wish to solve, while being both readable and easy to use by people with a basic background in robotic state estimation.

[1] Barfoot, T. D. and Furgale, P. T., “_Associating Uncertainty with Three-Dimensional Poses for use in Estimation Problems_,” IEEE Transactions on Robotics, 2014.

## Installation

### Hardware and Software Requirements

- Compiler with C++17 support
- Eigen (>=3.3.7)
- CMake or ROS2(colcon+ament_cmake)
- Boost library
- OpenMP
- [lgmath](https://github.com/utiasASRL/lgmath.git)

### Install c++ compiler, cmake, Boost and OpenMP

```bash
sudo apt -q -y install build-essential cmake libboost-all-dev libomp-dev
```

### Install Eigen (>=3.3.7)

Eigen can be installed using APT

```bash
sudo apt -q -y install libeigen3-dev
```

If installed from source to a custom location then make sure `cmake` can find it.

### Build and install using `cmake`

Install [lgmath](https://github.com/utiasASRL/lgmath.git) using `cmake` before installing this library. If installed to a custom location then make sure `cmake` can find it.

Clone this repo

```bash
git clone https://github.com/utiasASRL/steam.git .
```

Preprocessor macros

- `STEAM_DEFAULT_NUM_OPENMP_THREADS=<num. threads>`: Default to 4. Define number of threads to be used by OpenMP in STEAM.
- `STEAM_USE_OBJECT_POOL`: Default to undefined. If defined then STEAM will use an object pool to improve performance but is no longer thread safe.

Build and install

```bash
mkdir -p build && cd $_
cmake ..
cmake --build .
cmake --install . # (optional) install, default location is /usr/local/
```

Note: `steamConfig.cmake` will be generated in both `build/` and `<install prefix>/lib/cmake/steam/` to be included in other projects.

### Build and install using `ROS2(colcon+ament_cmake)`

Clone both lgmath and this repository in the same directory

```bash
git clone https://github.com/utiasASRL/lgmath.git
git clone https://github.com/utiasASRL/steam.git
```

Source your ROS2 workspace and then

```bash
colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON"
```

Same preprocessor macro mentioned above also apply.

## [License](./LICENSE)
