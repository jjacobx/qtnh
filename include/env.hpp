#ifndef ENV_HPP
#define ENV_HPP

namespace qtnh {
  struct QTNHEnv {
    int proc_id;
    int num_processes;
    int num_threads;

    QTNHEnv();
    ~QTNHEnv();

    void print() const;
  };
}

#endif