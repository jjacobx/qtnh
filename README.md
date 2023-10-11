# Generalised Tensor Network Hub

This project aims to create a generalised distributed software for performing contractions of tensor networks, for the purpose of simulating quantum circuits. 

To build and run the code, execute the following commands: 

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

examples/simple-tensor
```

Use `-DCMAKE_BUILD_TYPE=Debug` to print out more information at runtime. 


## Dependencies

 * [CMake](https://cmake.org/) – at least version 3.10
 * [Catch2](https://github.com/catchorg/Catch2) – at least version v3.10


## Tests

Use `make test` after building to run the tests. 

