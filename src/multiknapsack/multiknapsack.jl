module MultiDimensionalKnapsack

using LinearAlgebra
using Random
import Random.rand
using Distributions
using Statistics

using JuMP

using Gurobi

struct MultiKnapsackInstance
    m::Int  # number of knapsacks
    n::Int  # number of items

    p::Vector{Float32}  # value of each item
    b::Vector{Float32}  # capacity of each knapsack
    W::Matrix{Float32}  # weight matrix
end

function MultiKnapsackInstance(p, b, W)
    m, n = size(W)
    length(b) == m || error("Invalid size of b")
    length(p) == n || error("Invalid size of p")
    return MultiKnapsackInstance(m, n, p, b, W)
end

destructure(kp::MultiKnapsackInstance) = vcat(kp.p, kp.b, kp.W[:])

function build_mip(p::MultiKnapsackInstance; optimizer=Gurobi.Optimizer, binary=true)
    model = Model(optimizer)

    x = if binary
        @variable(model, x[1:p.n], Bin)
    else
        # Build continuous relaxation
        @variable(model, x[1:p.n], lower_bound=0, upper_bound=1)
    end
    @objective(model, Min, -dot(p.p, x))

    @constraint(model, knapsack, p.W*x .<= p.b)

    return model
end

"""
    MultiKnapsackGenerator

Instance generator for multi-dimensional knapsack problems.

See https://anl-ceeesa.github.io/MIPLearn/0.3/guide/problems/#Multi-Dimensional-Knapsack
"""
struct MultiKnapsackGenerator
    m::Int  # distribution of # of knapsacks
    n::Int  # distribution of # of items

    # Parameters of weight distribution
    wmin::Int
    wmax::Int
    # Correlation coeff
    kmin::Int
    kmax::Int
    # 
    umin::Float64
    umax::Float64
    #
    amin::Float64
    amax::Float64
end

function MultiKnapsackGenerator(m::Int, n::Int;
    wmin=0, wmax=1000,
    kmin=100, kmax=100,
    umin=0.0, umax=1.0,
    amin=0.25, amax=0.25,
)
    return MultiKnapsackGenerator(
        m, n,
        wmin, wmax,
        kmin, kmax,
        umin, umax,
        amin, amax,
    )
end

function Random.rand(rng::AbstractRNG, kg::MultiKnapsackGenerator, S::Int; as_array::Bool=false)
    m, n = kg.m, kg.n

    # Sample weights
    Ws = float.(rand(rng, kg.wmin:kg.wmax, m, n, S))

    # Sample correlation 
    K = kg.kmin .+ (kg.kmax - kg.kmin) .* rand(rng, S)
    # Sample profit multiplier
    u = kg.umin .+ (kg.umax - kg.umin) .* rand(rng, n, S)
    u = u * Diagonal(K)

    α = kg.amin .+ (kg.amax - kg.amin) .* rand(rng, m, S)

    # Generate knapsack capacities
    # b[i,s] = α[i,s] * Σⱼ w[i,j,s]
    bs = round.(α .* sum(Ws, dims=2)[:, 1, :])

    # Generate item prices
    ps = round.(u .+ mean(Ws, dims=1)[1, :, :])

    if as_array
        return ps, bs, Ws
    else
        return [
            MultiKnapsackInstance(m, n, ps[:, s], bs[:, s], Ws[:, :, s])
            for s in 1:S
        ]
    end
end

include("learning.jl")

end  # module