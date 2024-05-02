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

def gen_random_tn_header(tns : list[TensorNetwork], groups : list[(str, int)]):
  this_dir = os.path.dirname(os.path.realpath(__file__))
  with open(this_dir + '/random-tn.hpp', 'w', encoding = "utf-8") as f:
    f.write("#ifndef RANDOM_TN_HPP\n#define RANDOM_TN_HPP\n\n")
    f.write("#include \"contraction-validation.hpp\"\n\n")
    f.write("using namespace std::complex_literals;\n\n")

    f.write("namespace gen {\n")

    for i, tn in enumerate(tns) :
      f.write(f"  const tn_validation v{i+1} {{\n")
      f.write("    std::vector<tensor_info> {\n")
      for j, t in enumerate(tn.tensors):
        f.write("      tensor_info {{ ")
        for k, dim in enumerate(t.shape):
          f.write(f"{dim}")
          if (k < len(t.shape) - 1):
            f.write(", ")

        f.write(" }, { ")
        for k, el in enumerate(t.flatten()):
          plus = "+" if el.imag >= 0 else ""
          f.write(f"{el.real: .2f}{plus}{el.imag:.2f}i")
          if (k < t.size - 1):
            f.write(", ")

        f.write(" }}")
        if (j < len(tn.tensors) - 1):
          f.write(", \n")

      f.write("\n    }, \n\n")

      f.write("    std::vector<bond_info> {\n")
      for j, b in enumerate(tn.bonds):
        f.write(f"      bond_info {{ {b.i1}, {b.i2}, {{")
        for k, w in enumerate(b.ws):
          f.write(f"{{ {w[0]}, {w[1]} }}")
          if (j < len(b.ws) - 1):
            f.write(", ")
        f.write("}}")
        if (j < len(tn.bonds) - 1):
          f.write(", \n")
      f.write("\n    }, \n\n")

      res = contract(tn)
      f.write("    tensor_info {{ ")
      for k, dim in enumerate(res.shape):
        f.write(f"{dim}")
        if (k < len(res.shape) - 1):
          f.write(", ")

      f.write(" }, { ")
      for k, el in enumerate(res.flatten()):
        plus = "+" if el.imag >= 0 else ""
        f.write(f"{el.real: .2f}{plus}{el.imag:.2f}i")
        if (k < res.size - 1):
          f.write(", ")

      f.write(" }}\n")
      f.write("  };\n\n")


    k = 1
    for (name, n) in groups:
      f.write(f"  const std::vector<tn_validation> {name} {{ ")
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

  tns = []

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
  
  groups = [("tn2_vals", 5)]
  gen_random_tn_header(tns, groups)

if __name__ == "__main__":
  main()
  