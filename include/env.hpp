#include <complex>
#include <map>
#include <vector>

namespace qtnh {
  typedef std::size_t tidx;
  typedef std::vector<qtnh::tidx> tidx_tup;
  typedef unsigned short int tidx_tup_st;

  enum class TIdxFlag { open, closed, oob };

  typedef std::vector<TIdxFlag> tidx_flags;
  typedef std::complex<double> tel;
  typedef std::map<qtnh::tidx_tup, int> dmap;

  typedef std::pair<tidx, tidx> wire;

  struct QTNHEnv {
    int proc_id;
    int num_processes;
    int num_threads;

    QTNHEnv();
    ~QTNHEnv();

    void print() const;
  };
}
