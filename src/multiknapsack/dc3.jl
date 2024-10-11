"""
    initialize_fcnn_dc3(m, n)

Initialize a feedforward neural network for the DC3 setting.

The FCNN takes as input a vector of size `m+n+(m*n)` and outputs a vector of size `m+n`.
    The output vector represents the concatenation of `y` and `zl`.
"""
function initialize_fcnn_dc3(m, n)
    # Input size is m+n+(m*n)
    input_size = m + n + m*n
    hidden_size = 4*(m+n)
    model = Chain(
        Dense(input_size => hidden_size, sigmoid),
        Dense(hidden_size => hidden_size, sigmoid),
        Dense(hidden_size => m+n),
    )

    return model
end

"""
    dual_equality_completion_dc3(P, B, W, Y, Zl)

Complete the partial dual solution `y, zl` by recovering `Zu = W'y + zl + p`.
"""
function dual_equality_completion_dc3(P, B, W, Y, Zl)
    # We have W'y + zˡ - zᵘ = -p
    # The dual dual completion strategy reads `zᵘ = W'y + zˡ + p`
    # It only ensures equality constraints, not non-negativity of zᵘ
    Zu = batched_vec(batched_transpose(W), Y) + P + Zl
    return Zu
end

function compute_dual_inequality_violations_dc3(y, zl, zu)
    m = size(y, 2)
    # We want `y ≤ 0`, `zl ≥ 0`, `zu ≥ 0`
    δy  = sum(abs2, relu.(  y), dims=1)
    δzl = sum(abs2, relu.(-zl), dims=1)
    δzu = sum(abs2, relu.(-zu), dims=1)
    return δy, δzl, δzu
end

function dual_inequality_violation_grad_dc3(p, b, W, y, zl)
    # First recover `zu` from `y, zl`
    zu = dual_equality_completion_dc3(p, b, W, y, zl)

    # Gradient w.r.t `y`
    # ∇(relu(y)^2) / ∂y = 2 * relu(y) 
    # Plus a W*∇zu contribution for the zu violations
    ∇y  =  2 .* (relu.(y) - batched_vec(W, relu.(-zu)))

    # Gradient w.r.t `zl`
    # Includes gradient of violation w.r.t `zl >= 0` and `zu >= 0`
    ∇zl = -2 .* (relu.(-zl) + relu.(-zu))

    return ∇y, ∇zl
end

"""
    evaluate_model_dc3(dnn, x, p, b, W; dc3_corr_stepsize, dc3_corr_maxstep)

Evaluate DC3 model. Returns a full-space dual solution `y, zl, zu`.
    The returned solution will satisfy equality constraints, 
    but may violate dual inequality constraints `y ≤ 0, zl ≥ 0, zu ≥ 0`.
"""
function evaluate_model_dc3(dnn, x, p, b, W; 
    dc3_corr_stepsize=1f-4, 
    dc3_corr_maxstep=10,
    dc3_corr_tol=1f-2
)
    m, n, S = size(W)
    yz = dnn(vcat(p, b, flatten(W)))
    y = yz[1:m, :]
    zl = yz[(m+1):(m+n), :]

    # Take a few gradient steps
    for k in 1:dc3_corr_maxstep
        # DC3 needs to compute gradients of inequality violations w.r.t `y, zl`
        #   which includes the implicit computation of `zu`.
        # Since Zygote seems to have issues computing (some?) gradients inside a `withgradient` block,
        #   we need to compute the gradients manually.
        ∇y, ∇zl = dual_inequality_violation_grad_dc3(p, b, W, y, zl)

        # Large violations --> keep correcting
        # ⚠ do not use in-place operations because Zygote does not support mutating arrays
        y  = y  - dc3_corr_stepsize * ∇y
        zl = zl - dc3_corr_stepsize * ∇zl
    end

    # Re-run dual equality completion to make sure we have full solution
    zu = dual_equality_completion_dc3(p, b, W, y, zl)

    return y, zl, zu
end

function dc3_soft_loss(p, b, W, y, zl, zu; dc3_loss_rho=10f0)
    # Dual objective value b'y - e'zu
    bty = batched_vec(reshape(b, (1, size(b, 1), size(b, 2))), y)
    z_dual = -(bty - sum(zu, dims=1)) # ⚠ negate the dual objective since Flux _minimizes_ the loss

    δy, δzl, δzu = compute_dual_inequality_violations_dc3(y, zl, zu)

    return z_dual + dc3_loss_rho * δy + dc3_loss_rho * δzl + dc3_loss_rho * δzu
end
