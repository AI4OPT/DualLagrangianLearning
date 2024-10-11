isinteractive() && (using Revise)

using Base.Threads
using BenchmarkTools
using Printf
using Random
using Statistics

using CUDA
using Flux
using MLUtils
using JLD2

using DualLagrangianLearning
const DLL = DualLagrangianLearning
const PP = DLL.ProductionPlanning

using JuMP
using MosekTools, Mosek

function solve_dataset(dataset::PP.ProductionPlanningDataset; time_limit=60.0)
    D = dataset.D
    E = dataset.E
    R = dataset.R
    B = dataset.B

    n, S = size(D)

    Xopt = zeros(n, S)
    Yopt = zeros(1, S)
    Zopt = zeros(S)
    Lopt = zeros(S)
    ost = Vector{String}(undef, S)
    pst = Vector{String}(undef, S)
    dst = Vector{String}(undef, S)
    topt = zeros(S)

    # Solve all optimization problems and recover dual solution
    @threads for i in 1:S
        pp = PP.ProductionPlanningInstance(n, D[:, i], E[:, i], R[:, i], B[1, i])
        model = PP.build_rsoc(pp)
        set_silent(model)
        set_optimizer_attribute(model, "MSK_IPAR_NUM_THREADS", 1)
        set_time_limit_sec(model, time_limit)
        topt[i] = @elapsed optimize!(model)
        Xopt[:, i] .= value.(model[:x])
        y_opt = dual(model[:resource])
        ost[i] = string(termination_status(model))
        pst[i] = string(primal_status(model))
        dst[i] = string(dual_status(model))
        Yopt[1, i] = y_opt
        Zopt[i] = objective_value(model)
        Lopt[i] = PP.lagrangian(pp.d, pp.e, pp.r, pp.b, y_opt)
    end

    return Xopt, Yopt, Zopt, Lopt, ost, pst, dst, topt
end

function evaluate_dataset(dnn, dataset)
    X = vcat(dataset.D, dataset.E, dataset.R, dataset.B)
    ξ = dnn(X)
    Y = min.(0, ξ[1:1, :])  # ensure all the Ys are negative
    L = PP.lagrangian(dataset.D, dataset.E, dataset.R, dataset.B, Y)
    return Y, L
end

function profile_dnn_inference(dnn, D; dc3, dc3_corr_maxstep, dc3_corr_stepsize)
    # Make sure everything is on the GPU
    D, E, R, B = gpu(D.D), gpu(D.E), gpu(D.R), gpu(D.B)
    X = vcat(D, E, R, B)
    dnn = gpu(dnn)

    if dc3
        println("DC3 inference benchmark")
        b = @benchmark CUDA.@sync PP.evaluate_model_dc3($dnn, $D, $E, $R, $B, $X; dc3_corr_maxstep=$dc3_corr_maxstep, dc3_corr_stepsize=$dc3_corr_stepsize)
        return b
    else
        println("DLL inference benchmark")
        b = @benchmark CUDA.@sync PP.evaluate_model_dll($dnn, $D, $E, $R, $B, $X)
        return b
    end
end

function print_result_table(D)
    # Header
    @printf "%4s %6s %6s %6s %6s %6s\n" "n" "data" "avg" "std" "min" "max"

    # Main body
    for s in ["train", "val", "test"]
        dataset = D[s]["data"]
        n = dataset.n

        @printf "%4d %6s %4s" n s ""
        for k in ["avg", "std", "min", "max"]
            v = D[s]["ml"]["gap_$k"]
            @printf " %6.2f" (100*v)
        end
        @printf "\n"
    end
    return nothing
end

function main(n;
    dummy=false,
    N_train=8192,
    N_val=4096,
    N_test=4096,
    seed=42,
    # ML parameters
    η0=1e-4,
    ηmin=1e-7,
    minibatch_size=128,
    max_epoch=256,
    max_epoch_no_progress=64,
    min_epoch_before_patience=0,
    freq_log=16,
    time_limit=Inf,
    # DC3 parameters
    dc3=false,  # whether to use DC3 during training
    dc3_corr_maxstep=10,
    dc3_corr_tol=0.f0,
    dc3_corr_stepsize=1f-4,
    dc3_loss_rho=10f0,
)
    D = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "seed" => seed,
            "n" => n,
        )
    )
    # Generate training/validation/test data
    @info "Generating production planning instance data with n=$n items"
    rng = MersenneTwister(seed)
    ppg = PP.ProductionPlanningGenerator(n);
    dataset_train = rand(rng, ppg, N_train)
    dataset_val   = rand(rng, ppg, N_val)
    dataset_test  = rand(rng, ppg, N_test)

    # Keep track of everything
    D["train"] = Dict{String,Any}("data" => dataset_train)
    D["val"] = Dict{String,Any}("data" => dataset_val)
    D["test"] = Dict{String,Any}("data" => dataset_test)

    # Get ground truth values for all train/validation/test sets
    @info "Solving train/validation/test sets"
    for s in ["train", "val", "test"]
        dataset = D[s]["data"]
        Xopt, Yopt, Zopt, Lopt, ost, pst, dst, topt = solve_dataset(dataset)
        D[s]["opt"] = Dict{String,Any}(
            "y" => Yopt,
            "x" => Xopt,
            "obj_lp" => Zopt,
            "obj_lagrange" => Lopt,
            "termination_status" => ost,
            "primal_status" => pst,
            "dual_status" => dst,
            "time_opt" => topt,
        )
        freq_log > 0 && @printf("Mean LG bound (%5s): %+12.5e\n", s, mean(Lopt))
    end
    

    # Initialize DNN model
    # we always initialize a DNN with variable bounds
    # The training loop automatically gets rid of extra DNN outputs
    @info "Initializing the DNN"
    # ⚠ we re-seed the global RNG in Random to ensure we initialize the same FCNN
    Random.seed!(seed)
    dnn = gpu(PP.initialize_fcnn(n; dc3))

    # Train the DNN model
    @info "Training..."
    ttrain = @elapsed best_state = PP.train!(dnn, gpu(dataset_train), gpu(dataset_val);
        minibatch_size=minibatch_size,
        η0=η0,
        ηmin=ηmin,
        max_epoch=max_epoch,
        max_epoch_no_progress=max_epoch_no_progress,
        min_epoch_before_patience=min_epoch_before_patience,
        freq_log=freq_log,
        time_limit=time_limit,
        # DC3 parameters
        dc3=dc3,
        dc3_corr_maxstep=dc3_corr_maxstep,
        dc3_corr_stepsize=dc3_corr_stepsize,
        dc3_corr_tol=dc3_corr_tol,
        dc3_loss_rho=dc3_loss_rho,
    )
    freq_log > 0 && @printf("ML training time: %.3f\n", ttrain)
    D["best_ML_state"] = cpu(best_state)

    # Recover DNN with best validation loss
    # Move everything to CPU to make ML vs OPT comparison easier
    dnn_best = PP.initialize_fcnn(n; dc3) |> cpu
    Flux.loadmodel!(dnn_best, D["best_ML_state"])

    @info "Computing train/val/test gaps"
    for (s, dataset) in [("train", dataset_train), ("val", dataset_val), ("test", dataset_test)]
        Y, L = evaluate_dataset(dnn_best, dataset)
        L = L[:]

        Lopt = D[s]["opt"]["obj_lagrange"]
        gap = (Lopt - L) ./ (1e-8 .+ abs.(Lopt))
        gap_avg = mean(gap)
        gap_std = std(gap)
        gap_min = minimum(gap)
        gap_max = maximum(gap)

        D[s]["ml"] = Dict{String,Any}(
            "y" => Y,
            "obj_lagrange" => L,
            "gap" => gap,
            "gap_avg" => gap_avg,
            "gap_std" => gap_std,
            "gap_min" => gap_min,
            "gap_max" => gap_max,
            "training_time" => ttrain,
        )
    end

    # Benchmark performance on test set
    if !dummy
        b = profile_dnn_inference(dnn_best, dataset_test; dc3, dc3_corr_maxstep, dc3_corr_stepsize)
        display(b)
        D["test"]["ml"]["inference_time_ms"] = median(b.times ./ 1e6)
    end

    return D
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    n = parse(Int, ARGS[1])
    s = parse(Int, ARGS[2])
    method = parse(Int, ENV["SLURM_NODEID"]) == 0 ? "DLL" : "DC3"
    resdir = ARGS[4]
    if !isdir(resdir)
        println("Invalid result directory: $(resdir)")
        exit(1)
    end

    dc3 = (method == "DC3")

    @info "Compilation run"
    main(5; dummy=true, N_train=128, N_val=64, N_test=64, max_epoch=4, minibatch_size=16, freq_log=0, dc3=dc3)

    @info "Actual run ($(method))"
    D = main(n; seed=s,
        # ML options
        η0=2e-4,
        ηmin=1e-7,
        minibatch_size=128,
        max_epoch=4096,
        max_epoch_no_progress=128,
        min_epoch_before_patience=1024,
        time_limit=1800.0,  # 30-minute max training time
        # DC3 options
        dc3=dc3,
        dc3_corr_maxstep=10,
        dc3_corr_stepsize=1e-5,
        dc3_loss_rho=10f0,
    )
    
    println("ML results:")
    print_result_table(D)  # for user convenience
    println()

    # Save results
    JLD2.save(joinpath(resdir, "rcprod_n$(n)_s$(s)_$(method).jld2"), D)

    exit(0)
end

function postprocess()
    # Load all results
    D = Dict(
        n => Dict(
            method => JLD2.load("data/rcprod/rcprod_n$(n)_s42_vb0_$(method).jld2")
            for method in ["DC3", "DLL"]
        )
        for n in [10, 20, 50, 100, 200, 500, 1000]
    )

    postprocess(D)

    return D
end

function postprocess(D)
    # Test gaps statistics
    println("Optimality gaps stats")
    for n in [10, 20, 50, 100, 200, 500, 1000]
        d = D[n]
        @printf("%4d & %8.1f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f \\\\\n",
            n,
            gmean(d["DLL"]["test"]["opt"]["obj_lagrange"]),
            100 * d["DC3"]["test"]["ml"]["gap_avg"],
            100 * d["DC3"]["test"]["ml"]["gap_std"],
            100 * d["DC3"]["test"]["ml"]["gap_max"],
            100 * d["DLL"]["test"]["ml"]["gap_avg"],
            100 * d["DLL"]["test"]["ml"]["gap_std"],
            100 * d["DLL"]["test"]["ml"]["gap_max"],
        )
    end
    println()

    # TODO: constraint violation stats

    # ML architecture stats
    println("ML architecture stats")
    for n in [10, 20, 50, 100, 200, 500, 1000]
        @printf "%4d" n

        # input size, output size for DC3
        dnn_dc3 = PP.initialize_fcnn_dc3(n)
        n_param_dc3 = length(Flux.destructure(dnn_dc3)[1])
        @printf " & \$3n+1\$ & \$n+1\$ & %8d" n_param_dc3

        # input size, output size for DLL
        dnn_dll = PP.initialize_fcnn_dll(n)
        n_param_dll = length(Flux.destructure(dnn_dll)[1])
        @printf " & \$3n+1\$ & \$n+1\$ & %8d" n_param_dll

        @printf " \\\\\n"
    end
    println()

    # Training and inference time stats
    println("Timing stats")
    for n in [10, 20, 50, 100, 200, 500, 1000]
        d = D[n]
        N = length(d["DLL"]["test"]["opt"]["time_opt"])
        t_opt = sum(d["DLL"]["test"]["opt"]["time_opt"])
        t_dc3 = d["DC3"]["test"]["ml"]["inference_time_ms"]
        t_dll = d["DLL"]["test"]["ml"]["inference_time_ms"]

        @printf("%4d & %8.1f CPU.s & %6.1f GPU.ms & %6.1f GPU.ms\\\\\n",
            n,
            t_opt,
            t_dc3,
            t_dll,
        )
    end
    println()

    return nothing
end

function parse_log(flog)
    lns = readlines(flog)

    I = Int[]
    Lt1 = Float64[]
    Lv1 = Float64[]
    Lt2 = Float64[]
    Lv2 = Float64[]

    for ln in lns
        # ln[1] == '[' || continue
        occursin('|', ln) || continue
        s = split(ln)
        push!(I, parse(Int, s[2]))
        push!(Lt1, parse(Float64, s[5]))
        push!(Lv1, parse(Float64, s[6]))
        push!(Lt2, parse(Float64, s[10]))
        push!(Lv2, parse(Float64, s[10]))
    end

    return I, Lt1, Lt2, Lv1, Lv2
end

function plot_lagrange()
    n = 100

    i = 1
    d = data.D[:, i]
    e = data.E[:, i]
    r = data.R[:, i]
    b = data.B[1, i]
    xu = data.Xu[:, i]
    tu = data.Tu[:, i]

    yopt = -127.51524949863528

    function zlg(d, e, r, b, y, p1, t1)
        τ = relu.(e)
        pi = relu.(d .- (y .* r))
        σ = -sqrt.(2 .* τ .* pi)

        p1 = max(p1, σ[1] * σ[1] / (2 * t1))
        pi[1] = p1
        τ[1] = t1

        zxu = relu.(y .* r .+ pi .- d)
        ztu = relu.(τ .- e)

        return b*y - sqrt(2) * sum(σ) - dot(xu, zxu) - dot(tu, ztu)
    end

    f(p, t) = zlg(d, e, r, b, yopt, p, t)

    ps = 1000:20:2000
    ts = 100:1:200

    X = repeat(reshape(ps, 1, :), length(ts), 1)
    Y = repeat(ts, 1, length(ps))
    Z = map(f, X, Y)
    p1 = contour(ps, ts, Z, fill=true)
    # p2 = contour(y, t, Z)
    savefig(p1, "lg2.png")
    p1 = surface(X, Y, Z)
    savefig(p1, "lg2_surf.png")

    return nothing
end
