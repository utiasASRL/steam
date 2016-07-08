# How to Install Steam

## Dependencies
1. __Eigen 3.2.2__ and above (c++ math library). If you are on Ubuntu 14.04, you will need to compile it from source:

  ```
wget http://bitbucket.org/eigen/eigen/get/3.2.5.tar.gz
tar zxvf 3.2.5.tar.gz
rm 3.2.5.tar.gz
cd eigen-eigen-bdd17ee3b1b3/
mkdir build && cd build
cmake ..
sudo make install
  ```
2. __lgmath__ (Lie group math library). Follow the procedure from utiasASRL privite repository [INSTALL.md](https://github.com/utiasASRL/lgmath/blob/develop/INSTALL.md).
 

## Compilation and Install
In your development folder,
```
mkdir steam && cd steam
git clone https://github.com/utiasASRL/steam.git src
mkdir build && cd build
cmake ../src
sudo make install
```

## Enable Unit Tests 
(Optional)

1. Open CMake App
1. Enable TESTS_ON
1. cd build && make

## Uninstall
(Optional)

```
cd build
sudo make uninstall
```
