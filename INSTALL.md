# Dependencies

## Eigen
In a folder for 3rd party dependencies,
```bash
wget http://bitbucket.org/eigen/eigen/get/3.2.5.tar.gz
tar zxvf 3.2.5.tar.gz
cd eigen-eigen-bdd17ee3b1b3/
mkdir build && cd build
cmake ..
sudo make install
```

## lgmath
Follow the install instructions [here](https://github.com/utiasASRL/lgmath/blob/develop/INSTALL.md)

# Build
In your development folder,
```bash
mkdir steam-ws && cd $_
git clone https://github.com/utiasASRL/steam.git
cd steam && git submodule update --init --remote
```

Using [catkin](https://github.com/ros/catkin) and [catkin tools](https://github.com/catkin/catkin_tools) (recommended)
```bash
cd deps/catkin && catkin build
cd ../.. && catkin build
```

Using CMake (manual)
```bash
cd .. && mkdir -p build/catkin_optional && cd $_
cmake ../../steam/deps/catkin/catkin_optional && make
cd ../.. && mkdir -p build/catch && cd $_
cmake ../../steam/deps/catkin/catch && make
cd ../.. && mkdir -p build/steam && cd $_
cmake ../../steam && make -j4
```

# CMake Build Options

1. In your steam build folder (`build/steam`[`/steam`])
1. Open CMake cache editor (`ccmake .` or `cmake-gui .`)

# Install (optional)

Since the catkin build produces a catkin workspace you can overlay, and the CMake build exports packageConfig.cmake files, it is unnecessary to install steam except in production environments. If you are really sure you need to install, you can use the following procedure.

Using catkin tools (recommended)
```bash
cd steam
catkin profile add --copy-active install
catkin profile set install
catkin config --install
catkin build
```

Using CMake (manual)
```bash
cd build/steam
sudo make install
```

# Uninstall (Optional)

If you have installed, and would like to uninstall,

Using catkin tools (recommended)
```bash
cd steam && catkin clean -i
```

Using CMake (manual)
```bash
cd build/steam && sudo make uninstall
```
