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

#### Accessors

There are multiple ways to access tensor elements. The `fetch(tidx_tup)` method uses MPI to synchronise an element with all ranks – it involves a collective and needs to be called on all ranks. A less expensive way is to use the `at(tidx_tup)` method, which assumes the element is available locally, and otherwise throws an error. This should be used in conjunction with the boolean `has(tidx_tup)` to avoid runtime exceptions. Finally, the square bracket operator can be used on a tensor to directly address the vector with local elements. It is not recommended to use it outside of core implementations. 

```c++
std::cout << tp->fetch({ 0, 0, 1 }) << "\n"; // (1,0) on all ranks
if (tp->has({ 1, 0, 1 })) 
  std::cout << tp->at({ 1, 0, 1 }) << "\n"; // (0,1) on rank 1
if (env.proc_id <= 1) 
  std::cout << (*tp)[2] << "\n"; // (2,0) on rank 0 and (0,2) on rank 1
```

For non-special types of tensors, the accessors can also be used to set elements. There is an additional function, `put(tidx_tup, tel)`, which is a collective method to update an individual element without checking ranks. The element passed needs to be the same on all ranks, otherwise its behaviour is undefined. 

```c++
tp->put({ 0, 0, 0 }, 1-1i); // Update global element using global value. 
if (tp->has({ 1, 0, 0 })) 
  tp->at({ 1, 0, 0 }) = 1+1i; // Update global element using local value. 
if (env.proc_id <= 1)
  (*tp)[0] = 0-1i; // Update local element using local value. 
```

#### Scatter and broadcast

The way tensors are distributed is controlled by two operations – **scatter** and **broadcast**. Scatter controls how many indices are distributed, i.e. identified by process rank, in contrast to local indices used to access local elements. Distributed dimensions can be assigned during tensor construction, or modified using the static `Tensor::rescatter(tptr, int)` function. The offset specifies how many dimensions to convert to the other type: positive values convert last first local to last distributed indices, while negative values convert last distributed to first local indices. 

```c++
tp = Tensor::rescatter(std::move(tp), 1); // dis_dims = { 2, 2 }, loc_dims = { 2 }
tp = Tensor::rescatter(std::move(tp), -1); // dis_dims = { 2 }, loc_dims = { 2, 2 }
```

Broadcast defines how tensors are *copied* throughout processes. It is handled by a helper `Tensor::Broadcaster` class, an instance of which is stored by each tensor. There are three parameters that control broadcasting: 

 * *stretch* – how many times each local tensor chunk is repeated on consecutive processes. E.g. `0123` stretched by 2 becomes `00112233`. 
 * *cycles* – how many times the global tensor structure is repeated on consecutive groups of processes. E.g. `0123` cycled by 2 becomes `01230123`. 
 * *offset* – identifies the first process on which the tensor is stored. Can be used to make multiple tensor operations independent of each other. 

In addition, the broadcaster is responsible for storing the QTNH environment in the tensor, identifying whether current process is in use with an `active` flag, and handling communication between a single tensor instance (so that collectives can only see one copy of the tensor at a time). The broadcast parameters can be passed as the last argument of any `make` constructor via `BcParams` struct, or modified in an existing tensor using the static `Tensor::rebcast(tptr, BcParams)` function. 

```c++
// Same tensor as before, but with distributed indices of '0011' on ranks from 1 to 4. 
tptr tp = DenseTensor::make(env, { 2 }, { 2, 2 }, std::move(els), { 2, 1, 1 });

// Rescatter back to default broadcast parameters. 
// Distributed indices of `01` on ranks 0 and 1. 
tp = Tensor::rescatter(std::move(tp), { 1, 1, 0 });

// Distributed indices of `01010101` on ranks 0 and 7. 
tp = Tensor::rescatter(std::move(tp), { 1, 4, 0 });
```

#### Contraction

Two tensors can be contracted using the static `Tensor::contract(tptr, tptr, vector<wire>)` function. A `wire` represents a pair of index locations to contract with each other. The main restriction is that contracted indices need to be of the same type, i.e. both distributed or both local. An empty vector of wires makes the contraction equivalent to a *tensor product*. By default, when contracting two dense tensors, open indices from the second tensor will be appended at the end of open indices from the first one (after the right index type). 

```c++
tptr tp0 = DenseTensor::make(env, {}, { 2 }, { 1, 0 }); // |0> single-qubit state 

// Tensor product between |0> states. Copy prevents |0> from being deleted. 
tptr phi = Tensor::contract(tp0->copy(), tp0->copy(), {}); // Phi = |00> state
phi = Tensor::contract(std::move(phi), tp0->copy(), {}); // Phi = |000> state

// Distribute first qubit. 
tps = Tensor::rescatter(std::move(phi), 1);

auto a = std::pow(2, -.5); // Hadamard gate element
tptr had = DenseTensor::make(env, {}, { 2, 2 }, { a, a, a, -a }); // H gate

phi = Tensor::contract(std::move(phi), had->copy(), {{ 2, 0 }}); // |00+> state

// Scatter H gate to contract with distributed index. 
had = Tensor::rescatter(std::move(had), 2);
phi = Tensor::contract(std::move(phi), had->copy(), {{ 0, 0 }}); // |+0+> state
```

In the examples above, only last distributed or local indices of the state are contracted. This is because if the first local index is contracted, it will be moved to the last position because of index replacement. This can be remedied by using `ConParams` in place of the vector of wires, which is a struct to store more advanced parameters for contraction. It accept wires and dimension replacement tuples, which indicate where the indices should end up in the result. Closed indices are ignored. 

```c++
// Create contraction parameters with a (1, 0) wire. 
ConParams params({ 1, 0 });

// X is a wildcard, defined to be UINT16_MAX. 
params.dimRepls1 = { 0, X, 2 };
params.dimRepls2 = { X, 1 };

// Gather H gate to contract with local index. 
had = Tensor::rescatter(std::move(had), -2);
phi = Tensor::contract(std::move(phi), had->copy(), params); // |+0+> state
```

Some tensor contractions have different default dimension replacement policies. For instance, symmetric tensors try to replace contracted indices. The `ConParams` struct is passed to contraction by reference, which means it can be used to determine the replacement policy used, if a custom one was not defined. 

Self-contractions (i.e. within a single tensor) are not yet supported. 


### Indexing

Indexing defines a coordinate system that can be used to iterate through tensor elements, and is implemented in a class `TIndexing`. It consists of dimensions (`tidx_tup`) and index flags (vector of `TIFlag`, each of which contains a string label and integer tag). Index flags are useful when iterating only certain dimensions (while keeping the others constant) and for specifying iteration order (from largest to smallest tag). 

```c++
tidx_tup dims = { 2, 3 };
std::vector<TIFlag> ifls = { { "default", 0 }, { "other", 0 } };
TIndexing ti(dims, ifls);

qtnh::tidx_tup idx = { 0, 0 };  // idx = { 0, 0 }
idx = ti.next(idx);             // idx = { 0, 1 }
idx = ti.next(idx, "other");    // idx = { 1, 1 }
idx = ti.next(idx, "other");    // error, idx[0] > (dims[0] - 1)
```

The value of the incremented `tidx_tup idx` must be such that `idx[i] < dims[i]` for a given `TIndexing`, otherwise an error is thrown. 

`TIndexing` also allows creating two iterators, one for tuples, and another for corresponding numeric indices. The latter is mostly useful for directly accessing local elements in implementations. 

```c++
tidx_tup dims = { 2, 3, 2 };
std::vector<TIFlag> ifls = { { "default", 1 }, { "other", 0 }, { "default", 0 } };
TIndexing ti(dims, ifls);

// { 0, 0, 0 } { 1, 0, 0 } { 0, 0, 1 } { 1, 0, 1 }
for (auto idx : ti.tup()) 
  std::cout << idx << " ";
std::cout << idx << "\n";

// 0 6 1 7
for (auto i : ti.num()) 
  std::cout << i << " ";
std::cout << idx << "\n";

// { 0, 0, 0 } { 0, 1, 0 }
for (auto idx : ti.tup("other")) 
  std::cout << idx << " ";
std::cout << idx << "\n";
```

### Tensor Network

Class `TensorNetwork` acts as a storage for tensors, and bonds between them. Internal struct `TensorNetwork::Bond` implements the bonds as a vector of wires and a pair of IDs of tensors to contract, assigned when tensors are added to the network. Tensors and bonds can be accessed using their IDs. It is also possible to contract two tensors with given IDs, or to contract the entire network into a single tensor. When contracting multiple tensors, a *contraction order* of bond IDs can be specified. 

Tensors can be created directly inside of the tensor network, which helps avoid the hustle with unique pointers. A pointer to the tensor can be acquired by using `tensor()` method. 

```c++
QTNHEnv env;
TensorNetwork tn;

std::vector<tel> els1 = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
std::vector<tel> els2 = { 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 };

auto tid1 = tn.make<DenseTensor>(env, tidx_tup {}, tidx_tup { 2, 2, 2 }, std::move(els1));
auto tid2 = tn.make<DenseTensor>(env, tidx_tup {}, tidx_tup { 4, 2 }, std::move(els2));
auto bid1 = tn.addBond(t1_id, t2_id, {{ 0, 1 }});

//  ...

auto tid_new = tn.contract(bid1); // contract single bond
auto tid_res = tn.contractAll(); // contract all bonds

auto* tf = tn.tensor(tid_res); // get pointer to final tensor
```
