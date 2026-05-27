using ITensors, ITensorMPS
using StatsBase: sample

A00(site) = op("Proj0", site)
A01(site) = 0.5 * (op("X", site) + op("iY", site))
A10(site) = 0.5 * (op("X", site) - op("iY", site))
A11(site) = op("Proj1", site)

single_qubit_gate(els, site) = 
  els[1] * A00(site) + els[2] * A01(site) + 
  els[3] * A10(site) + els[4] * A11(site)

function rand_gate(site)
  i = sample(1:3)

  if i == 1
    els = [1, -1im, -1im, 1] / sqrt(2)
  elseif i == 2
    els = [1, -1, 1, 1] / sqrt(2)
  elseif i == 3
    els = [1 / sqrt(2), -0.5 -0.5im, 0.5 - 0.5im, 1 / sqrt(2)]
  end

  single_qubit_gate(els, site)
end

fSim(site1, site2, phi) = 
  A00(site1) * A00(site2) - 
  1im * A01(site1) * A10(site2) - 
  1im * A10(site1) * A01(site2) + 
  exp(-1im * phi) * A11(site1) * A11(site2)

function rcs_layer(mps, sites, layer; chi=16)
  init_sites_a = [
    1, 4, 6, 9, 11, 13, 15, 17, 19, 21, 24, 26, 
    28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51
  ]

  init_sites_b = [
    2, 5, 7, 10, 12, 16, 18, 20, 22, 25, 
    27, 29, 31, 34, 36, 38, 42, 44, 48
  ]


  init_sites_c = [
    0, 3, 8, 17, 26, 35, 43, 49, 1, 6, 13, 21, 
    30, 39, 4, 11, 19, 28, 37, 45, 9, 23, 32
  ]

  n_sites_c = [
    3, 5, 7, 9, 9, 8, 6, 4, 5, 7, 8, 9, 
    9, 8, 7, 8, 9, 9, 8, 6, 8, 9, 9
  ]

  init_sites_d = [
    2, 7, 14, 22, 31, 5, 12, 20, 29, 38, 
    10, 18, 27, 36, 44, 16, 25, 34, 42, 48
  ]

  n_sites_d = [
    5, 7, 8, 9, 9, 7, 8, 9, 9, 8, 
    8, 9, 9, 8, 6, 9, 9, 8, 6, 4
  ]

  for site in sites
    mps = apply(rand_gate(site), mps)
  end

  if layer == 'A'
    for i in init_sites_a
      mps = apply(fSim(sites[i + 1], sites[i + 2], pi / 6), mps; maxdim=chi)
    end
  elseif layer == 'B'
    for i in init_sites_b
      mps = apply(fSim(sites[i + 1], sites[i + 2], pi / 6), mps; maxdim=chi)
    end
  elseif layer == 'C'
    for (i, j) in zip(init_sites_c, n_sites_c)
      mps = apply(fSim(sites[i + 1], sites[i + j], pi / 6), mps; maxdim=chi)
    end
  elseif layer == 'D'
    for (i, j) in zip(init_sites_d, n_sites_d)
      mps = apply(fSim(sites[i + 1], sites[i + j], pi / 6), mps; maxdim=chi)
    end
  end

  mps
end

function rcs(mps, sites; chi=16)
  pattern = "ABCDCDABABCDCDABABCD"

  for (i, layer) in enumerate(collect(pattern))
    println("Computing layer: ", i)
    mps = rcs_layer(mps, sites, layer; chi=chi)
  end

  mps
end

CHI = parse(Int, ARGS[1])
println("Chi = ", CHI)

N = 53
sites = siteinds("Qubit", N; conserve_parity=false)
mps = MPS(sites, "0")

mps = @time rcs(mps, sites; chi=CHI)

println("F = ", norm(mps) ^ 2)
@show mps
