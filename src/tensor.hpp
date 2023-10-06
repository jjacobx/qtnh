#include <complex>
#include <vector>

typedef std::complex<double> complex;

class Tensor {
public:
    Tensor();
    Tensor(std::vector<complex> data, std::vector<int> dims);

    std::vector<int> getDims();
    complex getEl(std::vector<int> coords);

private:
    std::vector<complex> data;
    std::vector<int> dims;
};

// typedef std::pair<std::reference_wrapper<Tensor>, std::reference_wrapper<Tensor>> BondTensors;

class Bond {
public:
    Bond(std::pair<Tensor, Tensor> tensors, std::pair<int, int> dims);

    std::pair<Tensor, Tensor> getTensors();
    std::pair<int, int> getDims();
    int getSize();

private:
    std::pair<Tensor, Tensor> tensors;
    std::pair<int, int> dims;
};

class TensorNetwork {
public:
    TensorNetwork(std::vector<Tensor> tensors, std::vector<Bond> bonds);

    Tensor contract();

private:
    std::vector<Tensor> tensors;
    std::vector<Bond> bonds;
};
