# Standard packages
using JLD2
using LinearAlgebra
using Printf
using ProgressMeter
using Random
using Statistics

# ML stuff
using CUDA
using Flux
using MLUtils
using NNlib
using Zygote

struct MultiKnapsackDataset{T,M<:AbstractMatrix{T},A<:AbstractArray{T,3}}
    m::Int  # number of knapsacks
    n::Int  # number of items

    P::M  # value of each item,
    B::M  # capacity of each knapsack
    W::A  # weight matrix
end

# This allows to move datasets between CPU and GPU
Flux.@functor MultiKnapsackDataset

function MultiKnapsackDataset(P, B, W)
    m, n, S = size(W)
    return MultiKnapsackDataset(m, n, P, B, W)
end

function MultiKnapsackDataset(kps::Vector{MultiKnapsackInstance})
    m = kps[1].m
    n = kps[1].n
    S = length(kps)
    P = Matrix{Float32}(undef, n, S)
    B = Matrix{Float32}(undef, m, S)
    W = Array{Float32,3}(undef, m, n, S)
    for (s, kp) in enumerate(kps)
        P[:, s] .= kp.p
        B[:, s] .= kp.b
        W[:, :, s] .= kp.W
    end
    return MultiKnapsackDataset(m, n, P, B, W)
end

"""
    negate(x)

Negation function. Returns minus `x`.
"""
negate(x) = -x

include("dll.jl")
include("dc3.jl")

function initialize_fcnn(m, n; dc3=false)
    if !dc3
        return initialize_fcnn_dll(m, n)
    else
        return initialize_fcnn_dc3(m, n)
    end
end

function lagrangian(p::AbstractVector, b::AbstractVector, W::AbstractMatrix, y::AbstractVector)
    z = -p - W'y
    zu = max.(0, -z)
    return b'y - sum(zu)
end

function lagrangian(P::AbstractMatrix, B::AbstractMatrix, W::AbstractArray{T,3}, Y::AbstractMatrix) where {T}
    m, n, S = size(W)
    zu = relu.(batched_vec(batched_transpose(W), Y) + P)
    bty = batched_vec(reshape(B, (1, m, S)), Y)
    return bty - sum(zu, dims=1)
end

function train!(model, Dtrain, Dval;
    minibatch_size=32,
    max_epoch=1024,
    freq_log=0,
    η0=1e-4,
    ηmin=1e-7,
    max_epoch_no_progress=32,
    # DC3 parameters
    dc3::Bool=false,
    dc3_corr_maxstep=10,
    dc3_corr_tol=0f0,
    dc3_corr_stepsize=1e-5,
    dc3_loss_rho=10f0,
)
    # Grab data and move it to GPU
    Dtrain = gpu(Dtrain)
    P_train = gpu(Dtrain.P)
    B_train = gpu(Dtrain.B)
    W_train = gpu(Dtrain.W)
    X_train = vcat(Dtrain.P, Dtrain.B, flatten(Dtrain.W))

    Dval = gpu(Dval)
    P_val = gpu(Dval.P)
    B_val = gpu(Dval.B)
    W_val = gpu(Dval.W)
    X_val = vcat(P_val, B_val, flatten(W_val))

    loader = Flux.DataLoader((X_train, P_train, B_train, W_train), batchsize=minibatch_size, shuffle=true)

    η = η0
    optim = Flux.setup(Flux.Adam(η), model)

    # TODO: use separate structs for DLL and DC3 models
    #   and dispatch on the `evaluate_model` and `loss_fn` functions
    eval_fn = dc3 ? (dnn, x, p, b, W) -> evaluate_model_dc3(dnn, x, p, b, W; dc3_corr_stepsize, dc3_corr_maxstep, dc3_corr_tol) : evaluate_model_dll
    loss_fn = dc3 ? (p, b, W, y, zl, zu) -> dc3_soft_loss(p, b, W, y, zl, zu; dc3_loss_rho) : dll_loss

    # Track best state and validation loss
    best_state    = Flux.state(model)
    best_val_loss = Inf32
    n_epoch_no_improvement = 0  # number of epochs without improvement in validation loss

    # Main training loop
    tstart = time()
    # TODO: report mean gap w.r.t optimum when ground truth solution is provided
    for epoch in 0:max_epoch
        # Compute validation loss
        y_val, zl_val, zu_val = eval_fn(model, X_val, P_val, B_val, W_val)
        L_val = loss_fn(P_val, B_val, W_val, y_val, zl_val, zu_val)
        L_val = mean(L_val)
        
        # Check for improvement in validation loss
        # TODO: add a minimum threshold to register an improvement
        if L_val < best_val_loss
            # New best!
            best_state    = Flux.state(model)
            best_val_loss = L_val
            n_epoch_no_improvement = 0
        else
            n_epoch_no_improvement += 1
        end

        # Patience
        if n_epoch_no_improvement >= max_epoch_no_progress
            # Slow progress --> decrease learning rate by a factor 10...
            # ... and reset model to best validation loss
            η = η / 2
            if η < ηmin
                # Stop training
                println("Slow progress and minimum learning rate reached; stopping")
                break
            else
                println("Slow progress; resetting model and reducing learning rate to $(η)")
                # Reset model, optimizer, and progress counter
                Flux.loadmodel!(model, best_state)
                optim = Flux.setup(Flux.Adam(η), model)
                n_epoch_no_improvement = 0
            end
        end

        if (freq_log > 0) && (epoch % freq_log == 0)
            # Evaluate full training loss
            y_train, zl_train, zu_train = eval_fn(model, X_train, P_train, B_train, W_train)
            L_train = loss_fn(P_train, B_train, W_train, y_train, zl_train, zu_train)
            L_train = mean(L_train)

            # Log current progress
            tnow = time()
            @printf("%5.0f [%5d / %5d] ZL_train: %+12.5e  ZL_val: %+12.5e  (best: %+12.5e)\n",
                tnow - tstart,
                epoch,
                max_epoch,
                L_train,
                L_val,
                best_val_loss,
            )
        end
        for (x, p, b, W) in loader
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model
                y, zl, zu = eval_fn(m, x, p, b, W)
                # Compute training loss
                L = loss_fn(p, b, W, y, zl, zu)

                mean(L)  # mean loss over minibatch
            end
            Flux.update!(optim, model, grads[1])
        end
    end

    return best_state
end