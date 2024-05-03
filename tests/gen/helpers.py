import numpy as np
import random
import string

from collections import namedtuple
from dataclasses import dataclass
import numpy.typing as npt


Contraction = namedtuple("Contraction", "t1 t2 t3 ws")
# TensorNetwork = namedtuple("TensorNetwork", "tensors bonds")

@dataclass
class Bond:
  i1: int
  i2: int
  ws: list[tuple[int, int]]

@dataclass
class TensorNetwork:
  tensors: list[npt.ArrayLike]
  bonds: list[Bond]

def random_dims(min_idxs : int, max_idxs : int, allowed_dims : list[int]):
  n = np.random.randint(min_idxs, max_idxs)
  dims = random.choices(allowed_dims, k = n)
  return tuple(dims)

def random_wires(dims1 : tuple[int, ...], dims2 : tuple[int, ...], nwires : int):
  w1 = random.sample(range(len(dims1)), nwires)
  w2 = random.sample(range(len(dims2)), nwires)

  ws = zip(w1, w2)
  return list(ws)

def invert(ws : list[tuple[int, int]]):
  return [(j, i) for i, j in ws]

def make_compatible(dims1 : tuple[int, ...], dims2 : tuple[int, ...], wires : list[tuple[int, int]]):
  dims1 = list(dims1)
  dims2 = list(dims2)

  for i1, i2 in wires:
    dims1[i1] = dims2[i2]

  return tuple(dims1), tuple(dims2)


def random_tensor(dims : tuple[int, ...], dp = 1):
  real = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp
  imag = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp

  return real + imag * 1j


def swap_tensor(n1 : int, n2 : int):
  swap = np.zeros((n1, n2, n2, n1))
  for i in range(n1):
    for j in range(n2):
      swap[i, j, j, i] = 1
  
  return swap


def id_tensor(dims : tuple[int, ...]):
  return np.eye(np.prod(dims)).reshape(dims + dims)  


def make_contraction(t1, t2, ws):
  letters = list(string.ascii_lowercase)

  t1_idxs = letters[0:len(t1.shape)]
  t2_idxs = letters[len(t1.shape):(len(t1.shape) + len(t2.shape))]

  for i1, i2 in ws:
    t2_idxs[i2] = t1_idxs[i1]

  con_string = ''.join(t1_idxs) + ',' + ''.join(t2_idxs)

  return Contraction(t1, t2, np.einsum(con_string, t1, t2), ws)


def make_swap(t1, i1 : int, i2 : int):
  i1, i2 = sorted((i1, i2))

  letters = list(string.ascii_lowercase)
  t1_idxs = letters[0:len(t1.shape)]
  t3_idxs = t1_idxs.copy()

  t3_idxs[i1], t3_idxs[i2] = t3_idxs[i2], t3_idxs[i1]

  swap = swap_tensor(t1.shape[i1], t1.shape[i2])

  con_string = ''.join(t1_idxs) + '->' + ''.join(t3_idxs)
  
  return Contraction(t1, swap, np.einsum(con_string, t1), [(i1, 0), (i2, 1)])


def make_random_id(t1, n : int):
  idxs = random.sample(list(np.arange(0, len(t1.shape))), n)
  idxs.sort()

  # Subset the shape of t1
  dims = tuple(map(t1.shape.__getitem__, idxs))

  ws = zip(idxs, list(np.arange(0, n)))

  return Contraction(t1, id_tensor(dims), t1, list(ws))
