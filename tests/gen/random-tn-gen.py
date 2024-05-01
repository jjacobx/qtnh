import numpy as np
import os

from helpers import *

def contract(tn: TensorNetwork):
  letters = list(string.ascii_lowercase)

  idxs = []
  low_bound = 0
  for t in tn.tensors:
    upp_bound = low_bound + len(t.shape)
    idxs.append(letters[low_bound:upp_bound])
    low_bound = upp_bound

  for b in tn.bonds:
    for w in b.ws:
      idxs[b.i2][w[1]] = idxs[b.i1][w[0]]

  con_string = ""
  for idx in idxs:
    con_string += ''.join(idx) + ','
  con_string = con_string[:-1]

  return np.einsum(con_string, *tn.tensors)

def main():
  np.random.seed(9457)
  random.seed(9457)

  tns = []
  res = []

  # networks with single contraction
  for i in range(5):
    n = i % 4
    
    dims1 = random_dims(n + 1, 5, [2])
    dims2 = random_dims(n + 1, 5, [2])
    ws = random_wires(dims1, dims2, n)

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    tensors = [random_tensor(dims1), random_tensor(dims2)]
    bonds = [Bond(0, 1, ws)]

    tn = TensorNetwork(tensors, bonds)

    tns.append(tn)
    res.append(contract(tn))

if __name__ == "__main__":
  main()
  