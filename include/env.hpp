#ifndef ENV_HPP
#define ENV_HPP

namespace qtnh {
  struct QTNHEnv {
    unsigned int proc_id;
    unsigned int num_processes;
    unsigned int num_threads;

    QTNHEnv();
    ~QTNHEnv();

    void print() const;
  };
}

#endif