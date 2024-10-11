function initialize_fcnn_dc3(n)
    # Input size is 3*n+1
    # Output is `3*n+1` and does not enforce bounds on anything
    # Equality completion will recover `τ` and `π`
    # Bounds and conic constraints will be enforced by the correction strategy
    # Note: the output would be `1 + 3*n` in case of explicit bounds on ``x, t``
    input_size = 3*n+1
    hidden_size = max(128, 4*n)
    model = Chain(
        Dense(input_size => hidden_size, sigmoid),
        Dense(hidden_size => hidden_size, sigmoid),
        Parallel(vcat,
            Dense(hidden_size => 1),  # y
            # Dense(hidden_size => n),  # π
            # Dense(hidden_size => n),  # τ
            Dense(hidden_size => n),  # σ
        )
    )

    return model
end

"""
    dual_equality_completion_dc3(D, E, R, B, y, σ)

Recover `π, σ` by completing dual equality constraints.

The dual equality completion uses `τ = e` and `π = d - ry`.
    Bounds and conic constraints may not be satisfied
"""
function dual_equality_completion_dc3(D, E, R, B, y, σ)
    n, M = size(D)

    # Recover τ = e
    τ = E

    # Recover π = d - ry
    π = D - R .* y

    return y, π, τ, σ
end

"""
    compute_dual_inequality_violations_dc3(y, π, τ, σ)

Compute itemized dual inequality violations in dual solution `y, π, τ, σ`.

The dual inequalities and corresponding violations are:
    * `y ≤ 0`: max(0, y)²
    * `π ≥ 0`: min(0, -π)²
    * `τ ≥ 0`: min(0, -τ)²
    * `2πτ ≥ σ²`: max(0, σ² - 2πτ)²
"""
function compute_dual_inequality_violations_dc3(y, π, τ, σ)
    m = size(y, 2)
    # We want `y ≤ 0`, `π ≥ 0`, `τ ≥ 0`, and `2πτ ≥ σ²`
    δy  = sum(abs2, relu.(y), dims=1)
    δπ  = sum(abs2, relu.(-π), dims=1)
    δτ  = sum(abs2, relu.(-τ), dims=1)
    # RSOC inequality violation
    δqr = sum(abs2, relu.(σ .* σ - 2 .* π .* τ), dims=1)
    return δy, δπ, δτ, δqr
end

function dual_inequality_violation_grad_dc3(D, E, R, B, y, σ)
    y, π, τ, σ = dual_equality_completion_dc3(D, E, R, B, y, σ)

    # Gradient of ϕy w.r.t y
    ∂ϕy_y = 2 .* relu.(y)
    # (no gradient w.r.t σ)

    # Gradient of ϕπ w.r.t y
    # Recall that πⱼ = dⱼ - rⱼ*y
    ∂ϕπ_y = sum(2 .* R .* relu.(-π), dims=1)
    # (no gradient w.r.t σ)

    # Gradient of ϕτ w.r.t y
    # --> this is zero since τ does not depend on y

    # Gradient of ϕqr w.r.t y, σ
    # The RSOC violations is given by relu(σ² - 2 .* π .* τ)²
    δqr = relu.(σ .* σ - 2 .* π .* τ)
    ∂ϕqr_y = 4 .* sum(R .* τ .* δqr, dims=1)
    ∂ϕqr_σ = 4 .* σ .* δqr

    # Accumulate all the gradients and divide by minibatch size
    ∇y = (∂ϕy_y + ∂ϕπ_y + ∂ϕqr_y)
    ∇σ = ∂ϕqr_σ

    return ∇y, ∇σ
end

"""
    evaluate_model_dc3(dnn, D, E, R, B, X; dc3_corr_stepsize, dc3_corr_maxstep)

Evaluate DC3 model. Returns a full-space dual solution `y, π, σ, τ`.
    The returned solution will satisfy equality constraints, 
    but may violate dual inequality constraints `y ≤ 0, π ≥ 0, 2πτ ≥ σ²`.
"""
function evaluate_model_dc3(dnn, D, E, R, B, X; 
    dc3_corr_stepsize=1f-5, 
    dc3_corr_maxstep=10,
    dc3_corr_tol=1f-2
)
    n, m = size(D)  # n is the number of items, m is the minibatch size
    ξ = dnn(X)
    y = ξ[1:1, :]
    σ = ξ[(1+1):(1+n), :]

    # Take a few gradient steps
    for k in 1:dc3_corr_maxstep
        # Gradients of 
        ∇y, ∇σ = dual_inequality_violation_grad_dc3(D, E, R, B, y, σ)
        # # un-comment the lines below to see the progress of constraint violations
        # y, π, τ, σ = dual_equality_completion_dc3(D, E, R, B, y, σ)
        # δy, δπ, δτ, δqr = compute_dual_inequality_violations_dc3(y, π, τ, σ)
        # @printf "%4d  %.4e  %.4e  %.4e  %.4e\n" k sum(δy) sum(δπ) sum(δτ) sum(δqr)

        # ⚠ do not use in-place operations because Zygote does not support mutating arrays
        y = y - dc3_corr_stepsize * ∇y
        σ = σ - dc3_corr_stepsize * ∇σ
    end

    # Re-run dual equality completion to make sure we have full solution
    y, π, τ, σ = dual_equality_completion_dc3(D, E, R, B, y, σ)

    return y, π, τ, σ
end

function dc3_soft_loss(D, E, R, B, y, π, τ, σ; dc3_loss_rho=10f0)
    # Negated dual objective value
    sqr2 = sqrt(2f0)
    z_dual_neg = -B .* y .+ sqr2 .* sum(σ, dims=1)

    δy, δπ, δτ, δqr = compute_dual_inequality_violations_dc3(y, π, τ, σ)

    return z_dual_neg + dc3_loss_rho .* (δy + δπ + δτ + δqr)
end
