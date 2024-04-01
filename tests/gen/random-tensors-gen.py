import numpy as np
import random
import string
import os

from collections import namedtuple


Contraction = namedtuple("Contraction", "t1 t2 t3 ws")


def random_dims_and_wires(nwires, max_idxs, allowed_dims):
  n1 = np.random.randint(nwires + 1, max_idxs)
  n2 = np.random.randint(nwires + 1, max_idxs)

  dims1 = random.choices(allowed_dims, k = n1)
  dims2 = random.choices(allowed_dims, k = n2)

  w1 = random.sample(list(np.arange(0, n1)), nwires)
  w2 = random.sample(list(np.arange(0, n2)), nwires)

  for i1, i2 in zip(w1, w2):
    dims1[i1] = dims2[i2]

  ws = zip(w1, w2)

  return tuple(dims1), tuple(dims2), list(ws)


def random_tensor(dims, dp = 1):
  real = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp
  imag = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp

  return real + imag * 1j


def make_contraction(t1, t2, ws):
  letters = list(string.ascii_lowercase)

  t1_idxs = letters[0:len(t1.shape)]
  t2_idxs = letters[len(t1.shape):(len(t1.shape) + len(t2.shape))]

  for i1, i2 in ws:
    t2_idxs[i2] = t1_idxs[i1]

  con_string = ''.join(t1_idxs) + ',' + ''.join(t2_idxs)

  return Contraction(t1, t2, np.einsum(con_string, t1, t2), ws)


def gen_random_tensors_header(contractions : list[Contraction], groups : list[(str, int)]):
  this_dir = os.path.dirname(os.path.realpath(__file__))
  with open(this_dir + '/random-tensors.hpp', 'w', encoding = "utf-8") as f:
    f.write("#ifndef RANDOM_TENSORS_HPP\n#define RANDOM_TENSORS_HPP\n\n")
    f.write("#include \"contraction-validation.hpp\"\n\n")
    f.write("using namespace std::complex_literals;\n\n")

    f.write("namespace gen {\n")

    for i, con in enumerate(contractions) :
      f.write(f"  const contraction_validation v{i+1} {{\n")
      for t in (con.t1, con.t2, con.t3):
        f.write("    tensor_info {{ ")
        for j, dim in enumerate(t.shape):
          f.write(f"{dim}")
          if (j < len(t.shape) - 1):
            f.write(", ")

        f.write(" }, { ")
        for j, el in enumerate(t.flatten()):
          plus = "+" if el.imag >= 0 else ""
          f.write(f"{el.real: .2f}{plus}{el.imag:.2f}i")
          if (j < t.size - 1):
            f.write(", ")

        f.write(" }}, \n")

      f.write("\n")

      f.write("    std::vector<qtnh::wire>{")
      for j, w in enumerate(con.ws):
        f.write(f"{{ {w[0]}, {w[1]} }}")
        if (j < len(con.ws) - 1):
          f.write(", ")
      f.write("}\n")
      f.write("  };\n\n")

    k = 1
    for (name, n) in groups:
      f.write(f"  const std::vector<contraction_validation> {name} {{ ")
      for i in range(n):
        f.write(f"v{i+k}")
        if i < n - 1:
          f.write(", ")
      f.write(" };\n")
      k += n

    f.write("}\n\n")
    f.write("#endif")


def main():
  np.random.seed(9457)
  random.seed(9457)

  cons = []

  # 2-dimensional indices
  for i in range(5):
    dims1, dims2, ws = random_dims_and_wires(i % 4, 5, [2])
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2 and 3-dimensional indices
  for i in range(5):
    dims1, dims2, ws = random_dims_and_wires(i % 3, 4, [2, 3])
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2, 3 and 4-dimensional indices
  for i in range(5):
    dims1, dims2, ws = random_dims_and_wires(i % 2, 3, [2, 3, 4])
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2, 3, 4 and 5-dimensional indices
  for i in range(5):
    dims1, dims2, ws = random_dims_and_wires(i % 2, 3, [2, 3, 4, 5])
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  

  groups = [("dense_vals", 20)]
  gen_random_tensors_header(cons, groups)


if __name__ == "__main__":
  main()
