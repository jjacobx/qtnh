import os

from helpers import *


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
    n = i % 4
    
    dims1 = random_dims(n + 1, 5, [2])
    dims2 = random_dims(n + 1, 5, [2])
    ws = random_wires(dims1, dims2, n)

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2 and 3-dimensional indices
  for i in range(5):
    n = i % 3
    
    dims1 = random_dims(n + 1, 4, [2, 3])
    dims2 = random_dims(n + 1, 4, [2, 3])
    ws = random_wires(dims1, dims2, n)

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2, 3 and 4-dimensional indices
  for i in range(5):
    n = i % 2
    
    dims1 = random_dims(n + 1, 3, [2, 3, 4])
    dims2 = random_dims(n + 1, 3, [2, 3, 4])
    ws = random_wires(dims1, dims2, n)

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  # 2, 3, 4 and 5-dimensional indices
  for i in range(5):
    n = i % 2
    
    dims1 = random_dims(n + 1, 3, [2, 3, 4, 5])
    dims2 = random_dims(n + 1, 3, [2, 3, 4, 5])
    ws = random_wires(dims1, dims2, n)

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  for i in range(5):
    dims = random_dims(2, 4, [2, 3, 4])
    i1, i2 = random.sample(range(len(dims)), 2)

    dims, _ = make_compatible(dims, (2, 2), [(i1, 0), (i2, 1)])
    con = make_swap(random_tensor(dims), i1, i2)

    cons.append(con)

  for i in range(5):
    dims = random_dims(2, 4, [2, 3])
    i1, i2 = random.sample(range(len(dims)), 2)

    dims, _ = make_compatible(dims, (3, 3), [(i1, 0), (i2, 1)])
    con = make_swap(random_tensor(dims), i1, i2)

    cons.append(con)

  for i in range(5):
    dims = random_dims(2, 3, [2, 3])
    i1, i2 = random.sample(range(len(dims)), 2)

    dims, _ = make_compatible(dims, (2, 3), [(i1, 0), (i2, 1)])
    con = make_swap(random_tensor(dims), i1, i2)

    cons.append(con)

  for i in range(5):
    dims = random_dims(i // 2 + 2, 5, [2, 3])
    con = make_random_id(random_tensor(dims), i // 2 + 1)
    cons.append(con)

  for i in range(5):
    dims = random_dims(2, 3, [2, 3])
    con = make_random_id(random_tensor(dims), 2)
    cons.append(con)

  for i in range(5):
    n = i % 3
    dims1 = random_dims(2, 4, [2, 3])
    dims2 = random_dims(2, 4, [2, 3])

    ws = random_wires(dims1, dims2, n)
    
    dims1 = (2, ) + dims1
    ws = [(i + 1, j) for i, j in ws]

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  for i in range(5):
    n = i % 3
    dims1 = random_dims(2, 4, [2, 3])
    dims2 = random_dims(2, 4, [2, 3])

    ws = random_wires(dims1, dims2, n)
    
    dims1 = (3, ) + dims1
    ws = [(i + 1, j) for i, j in ws]

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    cons.append(con)

  for i in range(5):
    n = i % 3
    dims1 = random_dims(2, 4, [2, 3])
    dims2 = random_dims(2, 4, [2, 3])

    ws = random_wires(dims1, dims2, n)
    
    dims1 = (2, ) + dims1
    dims2 = (2, ) + dims2
    ws = [(i + 1, j + 1) for i, j in ws]

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    # Distributed index contraciton obeys different rules
    axes = np.arange(len(con.t3.shape))
    tax = len(dims1) - len(ws)
    axes = np.delete(np.insert(axes, 1, axes[tax]), tax + 1)

    con = con._replace(t3 = np.transpose(con.t3, axes))

    cons.append(con)

  for i in range(5):
    n = i % 3
    dims1 = random_dims(2, 3, [2, 3])
    dims2 = random_dims(2, 3, [2, 3])

    ws = random_wires(dims1, dims2, n)
    
    dims1 = (3, ) + dims1
    dims2 = (2, ) + dims2
    ws = [(i + 1, j + 1) for i, j in ws]

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    # Distributed index contraciton obeys different rules
    axes = np.arange(len(con.t3.shape))
    tax = len(dims1) - len(ws)
    axes = np.delete(np.insert(axes, 1, axes[tax]), tax + 1)

    con = con._replace(t3 = np.transpose(con.t3, axes))

    cons.append(con)

  for i in range(5):
    n = i % 3
    dims1 = random_dims(2, 3, [2, 3])
    dims2 = random_dims(2, 3, [2, 3])

    ws = random_wires(dims1, dims2, n)
    
    dims1 = (2, 2) + dims1
    dims2 = (2, ) + dims2
    ws = [(i + 2, j + 1) for i, j in ws]

    dims1, dims2 = make_compatible(dims1, dims2, ws)
    con = make_contraction(random_tensor(dims1), random_tensor(dims2), ws)

    # Distributed index contraciton obeys different rules
    axes = np.arange(len(con.t3.shape))
    tax = len(dims1) - len(ws)
    axes = np.delete(np.insert(axes, 2, axes[tax]), tax + 1)

    con = con._replace(t3 = np.transpose(con.t3, axes))

    cons.append(con)

  groups = [("dense_vals", 20), ("swap_vals", 10), ("invalid_swaps", 5), ("id_vals", 10), 
            ("mpi2r_vals", 5), ("mpi3r_vals", 5), ("mpi4r_vals", 5), ("mpi6r_vals", 5), ("mpi8r_vals", 5)]
  gen_random_tensors_header(cons, groups)


if __name__ == "__main__":
  main()
