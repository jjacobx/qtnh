# Quantum Tensor Network Hub

This project aims to create a generalised distributed software for contracting tensor networks, mainly for the purpose of quantum circuit simulations. 

To build and run the program, execute the following commands: 

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

mpirun -n 2 examples/qft
```

Use `-DCMAKE_BUILD_TYPE=Debug` to print out more information at runtime. 

Use `-DFSANITIZE=1` to detect memory leaks at runtime. 


## Dependencies

 * [CMake](https://cmake.org/) – at least version 3.13.0
 * [Catch2](https://github.com/catchorg/Catch2) – at least version v3.10
 * MPI - version 4.1


## Tests

Use `make test` after building to run the tests. This only comprises local/sequential tests. 

To run distributed MPI tests, use `mpirun -n <N> tests/c2-mpi "[<N>rank]"`, where `<N>` is the number of ranks to use (it can be 2 or 4). Use `--reporter root` to prevent duplicated output from all ranks. 


## Usage

As the first step before running any code, the environment `QTNHEnv` needs to be created. This is done by invoking an empty constructor. 

```c++
using namespace qtnh;

QTNHEnv env;
```

The environment initialises MPI, and can be used to access information such as the number of processes and current process ID. 

### Tensors

Currently, there are three main tensor classes with different properties: 

 * Dense tensors (`class DenseTensor`) – all tensor values can be non-zero, and there is no restriction to dimensions. 
 * Symmetric tensors (`class SymmTensor`) - all tensor values can be non-zero, but dimensions need to be split into equal input and output. This allows substituting contracted indices in-place. 
 * Diagonal tensors (`class DiagTensor`) - symmetric tensors where the only non-zero values appear on the input/output diagonal (i.e. when both are equal). 

All tensor classes are derived from a base class `Tensor`. There are also other special types of tensors, such as swap tensors or identity tensors. 

Tensors have no public constructors. Instead they need to be created using the static `make()` function, which is implemented for any non-virtual tensor class. It returns a unique pointer to the tensor, `std::unique_ptr<Tensor>` or `tptr`, which is the fundamental type used in the library. The reason for this is to ensure that tensors are not accidentally copied around, as they are likely to use a lot of memory. If a tensor has to be copied for some reason, there is an explicit `copy()` function, which returns a unique pointer to another tensor with the same class members. 

Tensor elements are of type `std::complex<double>` or `tel`, and their storage container is `std::vector`. The `make()` function accepts only rvalue references to elements, so if they are declared earlier, `std:move()` needs to be used (or a copy constructor). This is again to prevent wasting storage space. 

There are two dimension tuples used by tensors – distributed dimensions, and local dimensions, both stored as `std::vector<std::size_t>` or `tidx_tup`. The concatenation of both, in the same order, gives the actual total tensor dimensions. The distributed part is used to split the tensor into multiple ranks, along first `d` indices. Meanwhile, the remainder is used to address the elements locally. For instance, a rank 4 tensor might have dimensions `(2, 2, 2, 2)`, where the first `(2, 2)` are distributed, and the last `(2, 2)` are local. Then, the element `(1, 0, 0, 1)` will be available on rank `(1, 0)`, i.e. rank 2, and will be indexed by `(0, 1)`, i.e. position 1 in the local vector. 

With all that in mind, the code below shows how to construct a dense tensor. 

```c++
// Use this to type complex numbers. 
using namespace std::complex_literals;

std::vector<tel> els;
if (env.proc_id == 0) els = { 0+0i, 1+0i, 2+0i, 3+0i };
if (env.proc_id == 1) els = { 0+0i, 0+1i, 0+2i, 0+3i };

// Rank 3 tensor with dimensions (2, 2, 2), where (2) is distributed and (2, 2) local. 
tptr tp = DenseTensor::make(env, { 2 }, { 2, 2 }, std::move(els));
```

There are multiple ways to access tensor elements. The `fetch(tidx_tup)` method uses MPI to synchronise an element with all ranks – it involves a collective and needs to be called on all ranks. A less expensive way is to use the `at(tidx_tup)` method, which assumes the element is available locally, and otherwise throws an error. This should be used in conjunction with the boolean `has(tidx_tup)` to avoid runtime exceptions. Finally, the square bracket operator can be used on a tensor to directly address the vector with local elements. It is not recommended to use it outside of core implementations. 

```c++
std::cout << tp->fetch({ 0, 0, 1 }) << "\n"; // (1,0) on all ranks
if (tp->has({ 1, 0, 1 })) 
  std::cout << tp->at({ 1, 0, 1 }) << "\n"; // (0,1) on rank 1
if (env.proc_id <= 1) 
  std::cout << (*tp)[2] << "\n"; // (2,0) on rank 0 and (0,2) on rank 1
```


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

std::vector<qtnh::wire> ws = {{ 0, 1 }}; // connect index 0 of t1 and 1 of t2
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

