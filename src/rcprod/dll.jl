function initialize_fcnn_dll(n)
    # Input size is 3*n+1
    # Output is `1` (assuming "smart" dual recovery)
    # Note: the output would be `1 + 3*n` in case of explicit bounds on ``x, t``
    input_size = 3*n+1
    hidden_size = max(128, 4*n)
    model = Chain(
        Dense(input_size => hidden_size, sigmoid),
        Dense(hidden_size => hidden_size, sigmoid),
        Parallel(vcat,
            Chain(Dense(hidden_size => 1, softplus), negate),  # y
            # FIXME: we need these if we use variable bounds
            # Dense(hidden_size => n, softplus),  # π
            # Dense(hidden_size => n, softplus),  # τ
            # Dense(hidden_size => n),            # σ
        )
    )

    return model
end

"""
    dual_completion_dll(D, E, R, B, y)

Complete the partial dual solution `y` by recovering `π, τ, σ`; assumes `y ≤ 0`.

The returned solution `(y, π, τ, σ)` is dual-feasible.
"""
function dual_completion_dll(D, E, R, B, y)
    τ = E
    π = D - R .* y
    σ = - sqrt.(2 .* π .* τ)

    return y, π, τ, σ
end

function evaluate_model_dll(dnn, D, E, R, B, X)
    ξ = dnn(X)
    y, π, τ, σ = dual_completion_dll(D, E, R, B, ξ[1:1, :])
    return y, π, τ, σ
end

"""
    dll_loss(D, E, R, B, y, π, τ, σ)

The DLL self-supervised loss; same as negated Lagrangian.

See `lagrangian` function to compute the Lagrangian bound directly from `(D, E, R, B, y)`.
"""
function dll_loss(D, E, R, B, y, π, τ, σ)
    # ⚠ we NEGATE the dual bound since Flux _minimizes_ the loss
    # See `lagrangian` function to compute the Lagrangian bound directly
    sqr2 = sqrt(2f0)
    return  -B .* y .+ sqr2 .* sum(σ, dims=1)
end
