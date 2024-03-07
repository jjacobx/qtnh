# Quantum Tensor Network Hub

This project aims to create a generalised distributed software for performing contractions of tensor networks, for the purpose of simulating quantum circuits. 

To build and run the code, execute the following commands: 

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

mpirun -n 2 examples/qft
```

Use `-DCMAKE_BUILD_TYPE=Debug` to print out more information at runtime. 

Use `-DFSANITIZE=1` to detect memory leaks at runtime. 

Use `-DDEF_STENSOR_BCAST=0` to disable default shared tensor broadcasting. This can make the initialisation faster, but extra care needs to be taken when declaring shared objects. 


## Dependencies

 * [CMake](https://cmake.org/) – at least version 3.13.0
 * [Catch2](https://github.com/catchorg/Catch2) – at least version v3.10
 * MPI
 * OpenMP


## Tests

Use `make test` after building to run the tests. 


## Usage

The library uses two main custom types – `qtnh::tidx` for tensor indices (grouped into `qtnh::tidx_tup`), and `qtnh::tel` for tensor elements (collected in a vector `std::vector<qtnh::tel>`). Currently, the former is a wrapper around `std::size_t`, while the latter representes `std::complex<double>`. In addition, there is a custom tensor index type enum `qtnh::TIdxT`, which usually takes one of the two values `qtnh::TIdxT::open` or `qtnh::TIdxT:closed` for open and closed indices respectively. Paired together with an integer tag, it forms a flag `qtnh::tifl`, which labels tensor indices for contraction. To describe multiple indices, a vector of flags `qtnh::tifl_tup` is used. 

### Indexing

Indexing defines a coordinate system that can be used to iterate through tensor elements, and is implemented in a class `qtnh::TIndexing`. It consists of `qtnh::tidx_tup dims` to specify limits in each dimension, and `qtnh::tifl_tup ifls` to label each dimension of the tensor for contraction. The labels are pairs of index types (e.g. open or closed) and tags (useful when muliple indices are contracted). `qtnh::TIndexing` can be used to increment a `qtnh::tidx_tup` to address the next tensor element, while keeping either open or closed dimensions fixed: 

```c++
qtnh::tidx_tup dims = { 2, 3 };
qtnh::tifl_tup ifls = { { TIdxT::closed, 0 }, { TIdxT::open, 0 } };
TIndexing ti(dims, ifls);

qtnh::tidx_tup idx = { 0, 0 };      // idx = { 0, 0 }
idx = ti.next(idx, TIdxT::open);    // idx = { 0, 1 }
idx = ti.next(idx, TIdxT::closed);  // idx = { 1, 1 }
idx = ti.next(idx, TIdxT::closed);  // error, idx[0] > (dims[0] - 1)
```

The value of the incremented `qtnh::tidx_tup idx` must be such that `idx[i] < dims[i]` for a given `qtnh::TIndexing`, otherwise an error is thrown. 

`qtnh::TIndexing` also has a default iterator, which iterates through all the combinations of of open indices, while keeping the closed one fixed: 

```c++
qtnh::tidx_tup dims = { 2, 3, 2 };
qtnh::tifl_tup ifls = { { TIdxT::open, 0 }, { TIdxT::closed, 0 }, { TIdxT::open, 0 } };
TIndexing ti(dims, ifls);

// { 0, 0, 0 } { 0, 0, 1 } { 1, 0, 0 } { 1, 0, 1 }
for (auto idx : ti) {
    std::cout << idx << " ";
}
```

### Tensor

Tensors are multi-dimensional arrays of numbers defined in a base class `qtnh::Tensor`. Their dimensions are specified in `qtnh::tidx_tup dims`. A derived abstract class `qtnh::DenseTensor` generalises data storage to a vector of dense complex elements `std::vector<qtnh::tels> data`, and allows to rewrite all elements. Further distinction is made in classes `qtnh::SDenseTensor` and `qtnh::DDenseTensor`, where the former shares the same elements across all MPI ranks, while the latter distributes the first few dimensions to a necessary number of processes starting from rank 0 (ranks beyond maximum required contain empty data). To enable MPI, tensors need to be assigned a `qtnh::QTNHEnv`, which contains communication-related information. Finally, all tensors can be addressed using a `qtnh::tidx_tup idx` which corresponds to the index of the target element, while dense tensors can also be modified that way: 

```c++
QTNHEnv env;

qtnh::tidx_tup dims = { 2, 2 };
std::vector<qtnh::tel> data = { 0.0, 1.0, 1.0, 0.0 };
SDenseTensor X(env, dims, data); // X quantum gate

X.setLocEl({ 0, 0 }, X.getLocEl({ 0, 1 }).value()); // { 1.0, 1.0, 1.0, 0.0 };

// Print all tensor elements
TIndexing ti(dims); // by default all dimensions are open
for (auto idxs : ti) {
    std::cout << X.getLocEl(idxs).value() << " ";
}
```

There are three types of accessors with different behaviour when running on multiple processes: 
- global getters (`getDims()`, `getSize()`, `getEl(...)`) – read the tensor in its entirety
- local getters (`getLocDims()`, `getLocSize()`, `getLocEl(...)`) – read only local portion of the tensor
- distributed getters (`getDistDims()`, `getDistSize()`) – relate to how different local portions are distributed

The element accessors `getEl(...)` and `getLocEl(...)` are safe to use on all ranks, as their return type is optional. On the other hand, there is also an unsafe square bracket accessor `operator[...]`, which returns an error if the element is not available locally. The situation is similar for dense tensor setters – `setEL(...)` and `setLocEl(...)`, while the square bracket operator can be used to return reference to the local element.  

Tensors can be contracted with each other by using a static `qtnh::Tensor::contract(...)` method. It takes two unique tensor pointers, and a vector of *wires*, which are pairs of tensor indices to be contracted with each other. It is recommended to always use unique pointers when working with tensors, as the contraction arguments are deleted, which causes any related pointers to be invalidated. 

```c++
qtnh::QTNHEnv env;

qtnh::tidx_tup t1_dims = { 2, 2, 2 };
qtnh::tidx_tup t2_dims = { 4, 2 };

std::vector<qtnh::tel> t1_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
std::vector<qtnh::tel> t2_els = { 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 };

auto t1u = std::make_unique<SDenseTensor>(env, t1_dims, t1_els);
auto t2u = std::make_unique<SDenseTensor>(env, t2_dims, t2_els);

std::vector<qtnh::wire> ws = { { 0, 1 } }; // connect index 0 of t1 and 1 of t2
auto t3u = Tensor::contract(std::move(t1u), std::move(t2u), ws); // a (2, 2, 4) tensor
```

To distribute a shared tensor, use the `distribute(...)` function (unsafe option, as it returns a raw pointer), or contract its first indices with a `qtnh::ConvertTensor` (safe option). First `n` indices will be converted from local to distributed. It should be noted that distributed indices cannot be contracted, and instead need to be swapped with a shared one using a `swap(...)` function, or by using `qtnh::SwapTensor`. 

```c++
QTNHEnv env;

qtnh::tidx_tup t1_dims = { 2, 2, 2 };
std::vector<qtnh::tel> t1_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
auto t1u = std::make_unique<SDenseTensor>(env, t1_dims, t1_els);

// Distribute first index of t1. 
// Wrap result into unique pointer for safety. 
auto t2u = std::unique_ptr<DDenseTensor>(t1u->distribute(1));

// Or the safe option: 

// Create single-input convert tensor. 
auto cvu = std::make_unique<ConvertTensor>(env, qtnh::tidx_tup{ 2 });
// Connect it to first dimension of t1 and contract. 
auto t3u = Tensor::contract(std::move(t1u), std::move(cvu), {{ 0, 0 }});
// t1u is now invalid
```

### Tensor Network

Class `qtnh::TensorNetwork` acts as a storage for tensors, and bonds between them. Internal struct `qtnh::TensorNetwork::Bond` implements the bonds as a vector of wires and a pair of IDs of tensors to contract, assigned when tensors are added to the network. Tensors and bonds can be accessed using their IDs. It is also possible to contract two tensors with given IDs, or to contract the entire network into a single tensor. When contracting multiple tensors, a *contraction order* of bond IDs can be specified. 

Tensors can be created directly inside of the tensor network, which helps avoid the hastle with unique pointers. Reference to the tensor can be acquired by using `getTensor()` method. 

```c++
QTNHEnv env;
TensorNetwork tn;

qtnh::tidx_tup t1_dims = { 2, 2, 2 };
qtnh::tidx_tup t2_dims = { 4, 2 };

std::vector<qtnh::tel> t1_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
std::vector<qtnh::tel> t2_els = { 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 };

auto t1_id = tn.createTensor<SDenseTensor>(env, t1_dims, t1_els);
auto t2_id = tn.createTensor<SDenseTensor>(env, t2_dims, t2_els);
auto b1_id = tn.createBond(t1_id, t2_id, {{ 0, 1 }});

//  ...

auto new_id = tn.contract(b1_id); // contract single bond
auto final_id = tn.contractAll(); // contract all bonds

auto& tf = tn.getTensor(final_id); // get reference to final tensor
```

