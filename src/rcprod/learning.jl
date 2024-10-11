"""
    negate(x)

Negation function. Returns minus `x`.
"""
negate(x) = -x

function initialize_fcnn(n; dc3=false)
    if dc3
        return initialize_fcnn_dc3(n)
    else
        return initialize_fcnn_dll(n)
    end
end

# Note: the two functions below are equivalent, and coule be merged in a single non-dispatched function
# We keep both methods so that `lagrangian` returns a scalar when passed a `Vector` input
function lagrangian(d::AbstractVector, e::AbstractVector, r::AbstractVector, b::Real, y::Real)
    τ = e
    π = d - y .* r
    return b*y + 2 * sum(sqrt.(τ .* π))
end

function lagrangian(D::AbstractMatrix, E::AbstractMatrix, R::AbstractMatrix, B::AbstractMatrix, Y::AbstractMatrix)
    τ = E
    π = D - R .* Y
    return B .* Y .+ 2 * sum(sqrt.(τ .* π), dims=1)
end

include("dll.jl")
include("dc3.jl")

function train!(model, data_train, data_val;
    minibatch_size=32,
    max_epoch=1024,
    freq_log=0,
    η0=1e-4,
    ηmin=1e-7,
    max_epoch_no_progress=32,
    min_epoch_before_patience=0,
    time_limit=Inf,
    # DC3 parameters
    dc3=false,
    dc3_corr_maxstep=10,
    dc3_corr_tol=0.f0,
    dc3_corr_stepsize=1f-4,
    dc3_loss_rho=10f0,
)
    n = data_train.n
    # We assume that model and data are on the same device
    # and that we have enough GPU memory to store everything
    # --> no explicit GPU / CPU transfers here
    D_train = data_train.D
    E_train = data_train.E
    R_train = data_train.R
    B_train = data_train.B
    Xu_train = data_train.Xu
    Tu_train = data_train.Tu
    X_train = vcat(D_train, E_train, R_train, B_train)

    # Same with validation dataset
    D_val = data_val.D
    E_val = data_val.E
    R_val = data_val.R
    B_val = data_val.B
    Xu_val = data_val.Xu
    Tu_val = data_val.Tu
    X_val = vcat(D_val, E_val, R_val, B_val)

    loader = Flux.DataLoader(
        (X_train, D_train, E_train, R_train, B_train, Xu_train, Tu_train),
        batchsize=minibatch_size,
        shuffle=true
    )

    η = η0
    optim = Flux.setup(Flux.Adam(η), model)

    # TODO: use separate structs for DLL and DC3 models
    #   and dispatch on the `evaluate_model` and `loss_fn` functions
    eval_fn = dc3 ? (dnn, D, E, R, B, X) -> evaluate_model_dc3(dnn, D, E, R, B, X; dc3_corr_stepsize, dc3_corr_maxstep, dc3_corr_tol) : evaluate_model_dll
    loss_fn = dc3 ? (D, E, R, B, y, π, τ, σ) -> dc3_soft_loss(D, E, R, B, y, π, τ, σ; dc3_loss_rho) : dll_loss

    # Track best state and validation loss
    best_state    = Flux.state(model)
    best_val_loss = Inf32  # ⚠ we want to MAXIMIZE the Lagrange dual bound, so higher is better
    n_epoch_no_improvement = 0  # number of epochs without improvement in validation loss

    # Main training loop
    tstart = time()
    # TODO: report mean gap w.r.t optimum when ground truth solution is provided
    for epoch in 0:max_epoch
        # Compute validation loss
        y_val, π_val, τ_val, σ_val = eval_fn(model, D_val, E_val, R_val, B_val, X_val)
        L_val = loss_fn(D_val, E_val, R_val, B_val, y_val, π_val, τ_val, σ_val)
        L_val = mean(L_val)
        
        if L_val < best_val_loss
            # New best!
            best_state    = Flux.state(model)
            best_val_loss = L_val
            n_epoch_no_improvement = 0
        else
            n_epoch_no_improvement += 1
        end

        # Only count slow progress if at least a few epochs have elapsed
        (epoch <= min_epoch_before_patience ) && (n_epoch_no_improvement = 0)

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
            y_train, π_train, τ_train, σ_train = eval_fn(model, D_train, E_train, R_train, B_train, X_train)
            L_train = loss_fn(D_train, E_train, R_train, B_train, y_train, π_train, τ_train, σ_train)
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

        # Time limit
        if time() - tstart > time_limit
            println("Time limit reached; stopping")
            break
        end

        for (X, D, E, R, B, Xu, Tu) in loader
            loss, grads = Flux.withgradient(model) do dnn
                # TODO: re-instate variable bounds
                
                # Evaluate model
                y, π, τ, σ = eval_fn(dnn, D, E, R, B, X)
                # Compute training loss
                L = loss_fn(D, E, R, B, y, π, τ, σ)
                
                # Average loss over the minibatch
                mean(L)
            end
            Flux.update!(optim, model, grads[1])
        end
    end

    return best_state
end
