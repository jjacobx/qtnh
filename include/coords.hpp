#include <complex>
#include <memory>
#include <vector>

typedef std::vector<std::size_t> tidx_tuple;

enum class TIFlag { open, closed, oob };
typedef std::vector<TIFlag> tidx_flags;

std::ostream& operator<<(std::ostream& out, const tidx_tuple& o);

class TIndexing {
public:
    TIndexing();
    TIndexing(const tidx_tuple&);
    TIndexing(const tidx_tuple&, std::size_t);
    TIndexing(const tidx_tuple&, const tidx_flags&);

    const tidx_tuple& getDims() const;
    const tidx_flags& getFlags() const;

    bool isValid(const tidx_tuple&);
    bool isEqual(const tidx_tuple&, const tidx_tuple&, TIFlag = TIFlag::open);
    bool isLast(const tidx_tuple&, TIFlag = TIFlag::open);

    tidx_tuple& next(tidx_tuple&, TIFlag = TIFlag::open);
    tidx_tuple& prev(tidx_tuple&, TIFlag = TIFlag::open);
    tidx_tuple& reset(tidx_tuple&, TIFlag = TIFlag::open);

    TIndexing cut(TIFlag = TIFlag::open);

    class iterator {
    public:
        iterator(const tidx_tuple&, const tidx_flags&, const tidx_tuple&, TIFlag = TIFlag::open);

        const TIFlag& getActiveFlag() const;

        iterator& operator++();
        bool operator!=(const iterator&);
        tidx_tuple operator*() const;

    private:
        tidx_tuple dims;
        tidx_flags flags;

        tidx_tuple current;
        TIFlag active_flag;
    };

    iterator begin(TIFlag = TIFlag::open);
    iterator end();

    static TIndexing app(const TIndexing&, const TIndexing&);

private:
    tidx_tuple dims;
    tidx_flags flags;
};

