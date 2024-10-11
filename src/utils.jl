using HDF5
using JLD2
using JSON

# Geometric mean
gmean(u, s=zero(eltype(u))) = exp(mean(log.(u .+ s))) - s
gmean1(u) = gmean(u, one(eltype(u)))