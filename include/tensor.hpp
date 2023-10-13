#include <complex>
#include <vector>

typedef std::complex<double> complex;
typedef std::vector<std::size_t> dim_tuple;

enum class DimFlag { variable, fixed };
typedef std::vector<DimFlag> dim_flags;

class Coordinates {
public:
    Coordinates(dim_tuple limits);
    Coordinates(dim_tuple limits, std::size_t constant_dim);
    Coordinates(dim_tuple limits, dim_flags flags);
    Coordinates(dim_tuple limits, dim_flags flags, dim_tuple current);

    const dim_tuple& getLimits() const;
    const dim_flags& getFlags() const;
    const dim_tuple& getCurrent() const;
    void setCurrent(const dim_tuple& value);

    void next();
    // void operator++(int);
    void previous();
    // void operator--(int);
    void reset();

    bool canIncrease();
    bool canDecrease();

    Coordinates complement();
    Coordinates dropFixed();
    Coordinates operator&&(const Coordinates& rhs) const;

private:
    dim_tuple limits;
    dim_flags flags;
    dim_tuple current;
};

std::ostream& operator<<(std::ostream& out, const Coordinates& o);


class Tensor {
public:
    Tensor();
    Tensor(dim_tuple dims);
    Tensor(dim_tuple dims, std::vector<complex> data);

    const unsigned int& getID() const;

    const dim_tuple& getDims() const;
    const complex& getEl(const dim_tuple& coords) const;
    void setEl(const dim_tuple& coords, const complex& value);

    std::size_t size();

private:
    static unsigned int counter;
    unsigned int id;

    dim_tuple dims;
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
