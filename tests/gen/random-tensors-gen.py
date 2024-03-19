import numpy as np
import os

from collections import namedtuple

Contraction = namedtuple("Contraction", "t1 t2 t3 ws")

def random_tensor(dims, dp=1):
  real = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp
  imag = (np.random.randint(2 * 10**dp, size = dims) - 10**dp) / 10**dp

  return real + imag * 1j


def gen_random_tensors_header(contractions : list[Contraction]):
  this_dir = os.path.dirname(os.path.realpath(__file__))
  f = open(this_dir + '/random-tensors.hpp', 'w', encoding="utf-8")

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
      f.write(f"{{{w[0]}, {w[1]}}}")
      if (j < len(con.ws) - 1):
        f.write(", ")
    f.write("}\n")
    f.write("  };\n\n")

  f.write("  const std::vector<contraction_validation> cvs { ")
  for i in range(len(contractions)):
    f.write(f"v{i+1}")
    if i < len(contractions) - 1:
      f.print(", ")
  
  f.write(" };\n")

  f.write("}\n\n")
  f.write("#endif")


np.random.seed(9457)

t1 = random_tensor((2, 2))
t2 = random_tensor((2, 2))

con = Contraction(t1, t2, np.einsum("ij,ki->jk", t1, t2), [(0, 1)])
gen_random_tensors_header([con])
