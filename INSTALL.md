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
 

## Compilation
In your development folder,
```
git clone https://github.com/utiasASRL/steam.git
mkdir build && cd build
cmake ..
make
```

If you've installed a local version of Eigen, you will have to specify by changing the variable to something like this:
```
EIGEN3_INCLUDE_DIR=usr/local/include/eigen3 
```

## Unit Tests 
Just run
```
make test
```
to confirm that everything is alright.

## Installation
Once you validate that the unit tests pass, you can proceed to install the library:
```
sudo make install
```

The default location is `/usr/local/`.

## Uninstall

```
cd build
sudo make uninstall
```
