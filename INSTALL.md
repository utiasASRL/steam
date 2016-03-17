Dependencies
------------

First Install [lgmath and eigen](https://github.com/utiasASRL/lgmath)

# Install
In your development folder,
```
mkdir steam && cd steam
git clone https://github.com/utiasASRL/steam.git src
mkdir build && cd build
cmake ../src
sudo make install
```

# Enable Unit Tests 
(Optional)

1. Open CMake App
1. Enable TESTS_ON
1. cd build && make

# Uninstall
(Optional)

```
cd build
sudo make uninstall
```
