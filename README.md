# steam

STEAM (Simultaneous Trajectory Estimation and Mapping) Engine is an optimization library aimed at solving batch nonlinear optimization problems involving both SO(3)/SE(3) and continuous-time components. This is accomplished by using an iterative Gauss-Newton-style estimator in combination with techniques developed and used by ASRL. With respect to SO(3) and SE(3) components, we make use of the constraint sensitive perturbation schemes discussed in Barfoot and Furgale [1]. STEAM Engine is by no means intended to be the fastest car on the track; the intent is simply to be fast enough for the types of problems we wish to solve, while being both readable and easy to use by people with a basic background in robotic state estimation.

[1] Barfoot, T. D. and Furgale, P. T., “_Associating Uncertainty with Three-Dimensional Poses for use in Estimation Problems_,” IEEE Transactions on Robotics, 2014.

## Installation

### Dependencies

- Compiler with C++17 support and OpenMP
- CMake (>=3.16)
- Boost (>=1.71.0)
- Eigen (>=3.3.7)
- [lgmath (>=1.1.0)](https://github.com/utiasASRL/lgmath.git)
- (Optional) ROS2 Foxy or later (colcon+ament_cmake)

### Install c++ compiler, cmake and OpenMP

```bash
sudo apt -q -y install build-essential cmake libomp-dev
```

### Install Eigen (>=3.3.7)

```bash
# using APT
sudo apt -q -y install libeigen3-dev

# OR from source
WORKSPACE=~/workspace  # choose your own workspace directory
mkdir -p ${WORKSPACE}/eigen && cd $_
git clone https://gitlab.com/libeigen/eigen.git . && git checkout 3.3.7
mkdir build && cd $_
cmake .. && make install # default install location is /usr/local/
```

- Note: if installed from source to a custom location then make sure `cmake` can find it.

### Install lgmath

Follow the instructions [here](https://github.com/utiasASRL/lgmath.git).

### Build and install steam using `cmake`

```bash
WORKSPACE=~/workspace  # choose your own workspace directory
# clone
mkdir -p ${WORKSPACE}/steam && cd $_
git clone https://github.com/utiasASRL/steam.git .
# build and install
mkdir -p build && cd $_
cmake ..
cmake --build .
cmake --install . # (optional) install, default location is /usr/local/
```

Preprocessor macros

- `STEAM_DEFAULT_NUM_OPENMP_THREADS=<num. threads>`: Default to 4. Define number of threads to be used by OpenMP in STEAM.
- `STEAM_USE_OBJECT_POOL`: Default to undefined. If defined then STEAM will use an object pool to improve performance but is no longer thread safe.

Note: `steamConfig.cmake` will be generated in both `build/` and `<install prefix>/lib/cmake/steam/` to be included in other projects.

### Build examples

[samples/CMakeLists.txt](./samples/CMakeLists.txt) shows an example of how to add steam to your projects.

To build and run these samples:

```bash
cd ${WORKSPACE}/steam  ## $WORKSPACE defined above
mkdir -p build_samples && cd $_
cmake ../samples
cmake --build .  # Executables will be generated in build_samples
```

### Build and install steam using `ROS2(colcon+ament_cmake)`

```bash
WORKSPACE=~/workspace  # choose your own workspace directory

mkdir -p ${WORKSPACE}/steam && cd $_
git clone https://github.com/utiasASRL/steam.git .

source <your ROS2 worspace that includes steam>
colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON"
```

Same preprocessor macro mentioned above also apply.

## [License](./LICENSE)
