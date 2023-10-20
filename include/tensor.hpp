#include <complex>
#include <memory>
#include <vector>

#include "coords.hpp"

typedef std::complex<double> complex;

class Tensor {
public:
    Tensor();
    Tensor(tidx_tuple dims);
    Tensor(tidx_tuple dims, std::vector<complex> data);

    const unsigned int& getID() const;

    const tidx_tuple& getDims() const;
    const complex& getEl(const tidx_tuple& coords) const;
    void setEl(const tidx_tuple& coords, const complex& value);

    complex& operator[](tidx_tuple idx);

    std::size_t size();

private:
    static unsigned int counter;
    unsigned int id;

    tidx_tuple dims;
    std::vector<complex> data;
};

class Bond {
public:
    Bond(std::pair<Tensor, Tensor> tensors, std::pair<int, int> dims);

    int getID();

    std::pair<Tensor, Tensor> getTensors();
    std::pair<int, int> getDims();
    int getSize();

private:
    static unsigned int counter;
    int unsigned id;

    std::pair<Tensor, Tensor> tensors;
    std::pair<int, int> dims;
};

class TensorNetwork {
public:
    TensorNetwork(std::vector<Tensor> tensors, std::vector<Bond> bonds);

    Tensor getTensor(int n);
    void contract();

private:
    std::vector<Tensor> tensors;
    std::vector<Bond> bonds;
};
