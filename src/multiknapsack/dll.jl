"""
    initialize_fcnn_dll(m, n)

Initalize a feedforward neural network for the DLL setting.

The FCNN takes as input a vector of size `m+n+(m*n)` and outputs a scalar.
    The output represents the Lagrangian dual value `y`.
"""
function initialize_fcnn_dll(m, n)
    # Input size is m+n+(m*n)
    input_size = m + n + m*n
    hidden_size = 2*(m+n)
    model = Chain(
        Dense(input_size => hidden_size, sigmoid),
        Dense(hidden_size => hidden_size, sigmoid),
        Dense(hidden_size => m, softplus),
        negate,
    )

    #=
    Note: 
    It is possible to accept multiple inputs `p, b, W` using Flux's `Parallel` construct.
    In that case, one would need to call `dnn((p, b, W))`.
    This approach was found to be slower, so we keep the single-input approach here.
    =#

    return model
end

"""
    dual_completion_dll(P, B, W, Y)

Complete the partial dual solution `y` by recovering `Zl, Zu`.

The returned solution satisfies dual equality constraints and `zl, zu ≥ 0`.
The full solution (y, zl, zu) is dual-feasible if `y ≤ 0`.
"""
function dual_completion_dll(P, B, W, Y)
    # We have W'y + zˡ - zᵘ = -p
    # The dual dual completion strategy is to write
    # z = -p - W'y, then zˡ = z⁺, zᵘ = z⁻
    # This ensures equality constraints and non-negativity of zˡ, zᵘ
    Z = batched_vec(batched_transpose(W), Y) + P  # ⚠ this is -z
    Zu = relu.(Z)
    Zl = max.(0, -Z)
    return Y, Zl, Zu
end

function evaluate_model_dll(dnn, X, P, B, W)
    y = dnn(X)
    y, zl, zu = dual_completion_dll(P, B, W, y)
    return y, zl, zu
end

"""
    dll_loss(P, B, W, y, zl, zu)

The DLL self-supervised loss; same as negated Lagrangian.

See `lagrangian` function to compute the Lagrangian bound directly from `(P, B, W, y)`.
"""
function dll_loss(P, B, W, y, zl, zu)
    bty = batched_vec(reshape(B, (1, size(B, 1), size(B, 2))), y)
    # ⚠ we NEGATE the dual bound since Flux _minimizes_ the loss
    # See `lagrangian` function to compute the Lagrangian bound directly
    return  sum(zu, dims=1) - bty
end
