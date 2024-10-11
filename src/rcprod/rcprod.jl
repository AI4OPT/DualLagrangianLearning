module ProductionPlanning

using JLD2
using LinearAlgebra
using Printf
using ProgressMeter
using Random
import Random.rand
using Distributions
using Statistics

# ML stuff
using CUDA
using Flux
using MLUtils
using NNlib
using Zygote

import Flux:gpu, cpu

using JuMP

using Mosek
using MosekTools

struct ProductionPlanningInstance
    n::Int  # number of items

    d::Vector{Float32}  # inventory cost
    e::Vector{Float32}  # ordering cost
    r::Vector{Float32}  # resource usage of each item
    b::Float32          # resource capacity
end

function ProductionPlanningInstance(d, e, r, b)
    n = length(d)
    length(e) == n || error("Invalid size of e")
    length(r) == n || error("Invalid size of r")
    return ProductionPlanningInstance(n, d, e, r, b)
end

destructure(rcp::ProductionPlanningInstance) = vcat(rcp.d, rcp.e, rcp.r, rcp.b)

function build_rsoc(rcp::ProductionPlanningInstance; optimizer=Mosek.Optimizer)
    n = rcp.n
    d = rcp.d
    e = rcp.e
    r = rcp.r
    b = rcp.b

    model = Model(optimizer)

    @variable(model, x[1:n])
    @variable(model, t[1:n])

    @objective(model, Min, dot(d, x) + dot(e, t))

    # Resource constraint
    @constraint(model, resource, dot(r, x) <= b)

    # RSOC constraints
    sqr2 = sqrt(2)
    @constraint(model, rsoc[i in 1:n], [x[i], t[i], sqr2] in RotatedSecondOrderCone())

    return model
end

struct ProductionPlanningDataset{T,M<:AbstractMatrix{T}}
    n::Int  # number of items

    D::M  # linear cost, size n×N
    E::M  # inverse cost, size n×N
    R::M  # weight, size n×N
    B::M  # capacity, size 1×N
    Xu::M  # upper bound on X variables
    Tu::M  # upper bound on T variables
end

Flux.gpu(D::ProductionPlanningDataset) = ProductionPlanningDataset(D.n, Flux.gpu(D.D), Flux.gpu(D.E), Flux.gpu(D.R), Flux.gpu(D.B), Flux.gpu(D.Xu), Flux.gpu(D.Tu))
Flux.cpu(D::ProductionPlanningDataset) = ProductionPlanningDataset(D.n, Flux.cpu(D.D), Flux.cpu(D.E), Flux.cpu(D.R), Flux.cpu(D.B), Flux.gpu(D.Xu), Flux.gpu(D.Tu))

"""
ProductionPlanningGenerator

Instance generator for multi-dimensional knapsack problems.

See https://anl-ceeesa.github.io/MIPLearn/0.3/guide/problems/#Multi-Dimensional-Knapsack
"""
struct ProductionPlanningGenerator
    n::Int  # distribution of # of items

    # Demand of each item
    dmin::Float32
    dmax::Float32
    # Production costs
    cpmin::Float32
    cpmax::Float32
    # Rate of holding costs
    crmin::Float32
    crmax::Float32

    # Parameters for white noise
    amin::Float32
    amax::Float32
    bmin::Float32
    bmax::Float32
    emin::Float32
    emax::Float32
end

function ProductionPlanningGenerator(n::Int;
    dmin=1, dmax=100,
    cpmin=1, cpmax=10,
    crmin=0.05, crmax=2.0,
    amin=0.1, amax=1.5,
    bmin=0.1, bmax=2.0,
    emin=0.25, emax=0.75,
)
    return ProductionPlanningGenerator(
        n,
        dmin, dmax,
        cpmin, cpmax,
        crmin, crmax,
        amin, amax,
        bmin, bmax,
        emin, emax,
    )
end

function Random.rand(rng::AbstractRNG, ppg::ProductionPlanningGenerator, S::Int)
    n = ppg.n

    # Item demand
    Dem = ppg.dmin .+ (ppg.dmax - ppg.dmin) .* rand(rng, Float32, n, S)

    # Production costs
    Cp = ppg.cpmin .+ (ppg.cpmax - ppg.cpmin) .* rand(rng, Float32, n, S)
    # Rate of holding costs
    Cr = ppg.crmin .+ (ppg.crmax - ppg.crmin) .* rand(rng, Float32, n, S)
    
    # Ordering costs
    α = ppg.amin .+ (ppg.amax - ppg.amin) .* rand(rng, Float32, n, S)
    Co = α .* Cp

    # Storage requirements
    β = ppg.bmin .+ (ppg.bmax - ppg.bmin) .* rand(rng, Float32, n, S)
    R = β .* Cp

    # Storage capacity
    η = ppg.emin .+ (ppg.emax - ppg.emin) .* rand(rng, Float32, 1, S)
    B = η .* sum(R, dims=1)

    # Finally, convert to problem parameters
    D = (Cp .* Cr) ./ 2
    E = Co .* Dem

    Xu = zeros(Float32, n, S)
    Tu = zeros(Float32, n, S)
    # compute variable upper bounds
    for i in 1:S
        Xu[:, i] .= xu = B[1, i] ./ R[:, i]
        x_ = xu ./ n
        M = sum(D[:, i] .* x_) + sum(E[:, i] ./ x_)
        Tu[:, i] = M ./ E[:, i]
    end
    
    return ProductionPlanningDataset(n, D, E, R, B, Xu, Tu)
end

include("learning.jl")

end  # module