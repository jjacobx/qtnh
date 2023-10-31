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


## Usage

The library uses two main custom types – `tidx_tuple` for tensor indices, and `tels_array` for tensor elements. The former is implemented by a vector of non-negative integers, while the latter is a vector of complex numbers. In addition, there is a custom tensor index flag `TIFlag`, which usually takes one of the two values `TIFlag::open` or `TIFlag::closed` for open and closed indices respectively. To describe multiple dimensions, a vector of flags `tidx_flags` is used. 

### Indexing

Indexing defines a coordinate system that can be used to iterate through tensor elements, and is implemented in `class TIndexing`. It consists of `tidx_tuple dims` to specify limits in each dimension, and `tidx_flags flags` to indicate whether each of the dimensions is open or closed. `TIndexing` can be used to increment a `tidx_tuple` to address the next tensor element, while keeping either open or closed dimensions fixed: 

```c++
tidx_tuple dims = { 2, 3 };
tidx_flags flags = { TIFlag::closed, TIFlag::open };
TIndexing ti(dims, flags);

tidx_tuple idx = { 0, 0 };           // idx = { 0, 0 }
idx = ti.next(idx, TIFlag::open);    // idx = { 0, 1 }
idx = ti.next(idx, TIFlag::closed);  // idx = { 1, 1 }
idx = ti.next(idx, TIFlag::closed);  // error, idx[0] > (dims[0] - 1)
```

The value of the incremented `tidx_tuple idx` must be such that `idx[i] < dims[i]` for a given `TIndexing`, otherwise an error is thrown. 

`TIndexing` also has a default iterator, which iterates through all the combinations of of open indices, while keeping the closed one fixed: 

```c++
tidx_tuple dims = { 2, 3, 2 };
tidx_flags flags = { TIFlag::open, TIFlag::closed, TIFlag::open };
TIndexing ti(dims, flags);

for (auto idx : ti) {
    std::cout << idx << " "; // { 0, 0, 0 } { 0, 0, 1 } { 1, 0, 0 } { 1, 0, 1 },
}
```

### Tensor

Tensors are multi-dimensional arrays of numbers defined in `class Tensor`. Their dimensions are specified in `tidx_tuple dims`, and complex elements in `tels_array data`. Tensors can be addressed and modified using a `tidx_tuple idx` which corresponds to the index of the target element: 

```c++
tidx_tuple dims = { 2, 2 };
tels_array data = { 0.0, 1.0, 1.0, 0.0 };
Tensor X(dims, data); // X quantum gate

tidx_tuple idx1 = { 0, 0 };
tidx_tuple idx2 = { 0, 1 };
X[idx1] = X[idx2]; // { 1.0, 1.0, 1.0, 0.0 };

// Print all tensor elements
TIndexing ti(dims); // by default all dimensions are open
for (auto idx : ti) {
    std::cout << X[idx] << " ";
}
```
