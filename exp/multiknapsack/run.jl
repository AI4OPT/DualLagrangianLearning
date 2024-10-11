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
const MDK = DLL.MultiDimensionalKnapsack

using JuMP, Gurobi
const GRBENV = Gurobi.Env()

function solve_dataset(D::MDK.MultiKnapsackDataset; binary=false, time_limit=60.0)
    P = D.P
    B = D.B
    W = D.W

    m, n, S = size(W)
    Ygrb = zeros(m, S)
    Zgrb = zeros(S)
    Lgrb = zeros(S)
    Tgrb = zeros(S)

    # Solve all 
    @threads for i in 1:S
        kp = MDK.MultiKnapsackInstance(m, n, P[:, i], B[:, i], W[:, :, i])
        lp = MDK.build_mip(kp; optimizer=() -> Gurobi.Optimizer(GRBENV), binary=binary)
        set_silent(lp)
        set_optimizer_attribute(lp, "Threads", 1)
        set_time_limit_sec(lp, time_limit)
        optimize!(lp)
        y_grb = dual.(lp[:knapsack])
        Ygrb[:, i] .= y_grb
        Zgrb[i] = objective_value(lp)
        Lgrb[i] = MDK.lagrangian(kp.p, kp.b, kp.W, y_grb)
        Tgrb[i] = solve_time(lp)
    end

    return Ygrb, Zgrb, Lgrb, Tgrb
end

function evaluate_dataset(dnn, D)
    Y = dnn(vcat(D.P, D.B, flatten(D.W)))
    # Extract only the `Y` predictions, and retain only the negative part
    #   (we need y ≤ 0 for dual feasibility)
    Y = min.(0, Y[1:D.m, :])
    L = MDK.lagrangian(D.P, D.B, D.W, Y)[:]
    return Y, L
end

function print_result_table(D)
    # Header
    @printf "%4s %4s %6s %6s %6s %6s %6s\n" "m" "n" "data" "avg" "std" "min" "max"

    # Main body
    for s in ["train", "val", "test"]
        dataset = D[s]["data"]
        m, n = dataset.m, dataset.n

        @printf "%4d %4d %6s" m n s
        for k in ["avg", "std", "min", "max"]
            v = D[s]["ml"]["gap_$k"]
            @printf " %6.2f" (100*v)
        end
        @printf "\n"
    end
    return nothing
end

function profile_dnn_inference(dnn, D; dc3, dc3_corr_maxstep, dc3_corr_stepsize)
    # Make sure everything is on the GPU
    P, B, W = gpu(D.P), gpu(D.B), gpu(D.W)
    X = gpu(vcat(D.P, D.B, flatten(D.W)))
    dnn = gpu(dnn)

    b = if dc3
        println("DC3 inference benchmark")
        b = @benchmark CUDA.@sync MDK.evaluate_model_dc3($dnn, $X, $P, $B, $W; dc3_corr_maxstep=$dc3_corr_maxstep, dc3_corr_stepsize=$dc3_corr_stepsize)
        b
    else
        println("DLL inference benchmark")
        b = @benchmark CUDA.@sync MDK.evaluate_model_dll($dnn, $X, $P, $B, $W)
        b
    end

    display(b)

    return b
end

function postprocess()
    # Load all results
    D = Dict(
        (m, n) => Dict(
            method => JLD2.load("data/multiknapsack/mdk_m$(m)_n$(n)_s42_$(method).jld2")
            for method in ["DC3", "DLL"]
        )
        for m in [5, 10, 30], n in [100, 200, 500]
    )

    postprocess(D)

    return D
end

function postprocess(D)
    # Test gaps statistics
    println("Optimality gaps stats")
    for m in [5, 10, 30], n in [100, 200, 500]
        n == 100 && @printf("\\midrule\n")
        d = D[m, n]
        @printf("%4d & %4d & %8.1f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f & %6.2f \\\\\n",
            m, n,
            gmean(-d["DLL"]["test"]["gurobi"]["obj_lagrange"]),
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
    for m in [5, 10, 30], n in [100, 200, 500]
        @printf "%4d & %4d" m n

        # input size, output size for DC3
        dnn_dc3 = MDK.initialize_fcnn_dc3(m, n)
        n_param_dc3 = length(Flux.destructure(dnn_dc3)[1])
        @printf " & \$3n+1\$ & \$n+1\$ & %8d" n_param_dc3

        # input size, output size for DLL
        dnn_dll = MDK.initialize_fcnn_dll(m, n)
        n_param_dll = length(Flux.destructure(dnn_dll)[1])
        @printf " & \$3n+1\$ & \$n+1\$ & %8d" n_param_dll

        @printf " \\\\\n"
    end
    println()

    # Training and inference time stats
    println("Timing stats")
    for m in [5, 10, 30], n in [100, 200, 500]
        n == 100 && @printf("\\midrule\n")
        d = D[m, n]
        N = length(d["DLL"]["test"]["gurobi"]["solve_time"])
        t_opt = sum(d["DLL"]["test"]["gurobi"]["solve_time"])
        t_dc3 = d["DC3"]["test"]["ml"]["inference_time"]
        t_dll = d["DLL"]["test"]["ml"]["inference_time"]

        # Training times, in minutes
        t_tr_dc3 = get(d["DC3"]["test"]["ml"], "training_time", NaN) / 60
        t_tr_dll = get(d["DLL"]["test"]["ml"], "training_time", NaN) / 60

        @printf("%4d & %4d & -- & %8.1f CPU.s & %5.1f min & %6.1f GPU.ms & %5.1f min & %6.1f GPU.ms\\\\\n",
            m, n,
            t_opt,
            t_tr_dc3,
            t_dc3,
            t_tr_dll,
            t_dll,
        )
    end
    println()

    return D
end

function main(m, n;
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
    freq_log=16,
    # DC3 parameters
    dc3=false,  # whether to use DC3 during training
    dc3_corr_maxstep=10,
    dc3_corr_tol=0.f0,
    dc3_corr_stepsize=1f-3,
    dc3_loss_rho=10f0,
)
    dummy && @info "Dummy run"

    D = Dict{String,Any}(
        "meta" => Dict{String,Any}(
            "seed" => seed,
            "m" => m,
            "n" => n,
        )
    )
    # Generate training/validation/test data
    @info "Generating instance data with n=$n items and m=$m resources"
    rng = MersenneTwister(seed)
    kg = MDK.MultiKnapsackGenerator(m, n);
    D_train = MDK.MultiKnapsackDataset(rand(rng, kg, N_train; as_array=true)...)
    D_val   = MDK.MultiKnapsackDataset(rand(rng, kg, N_val; as_array=true)...)
    D_test  = MDK.MultiKnapsackDataset(rand(rng, kg, N_test; as_array=true)...)

    # Keep track of everything
    D["train"] = Dict{String,Any}("data" => D_train)
    D["val"] = Dict{String,Any}("data" => D_val)
    D["test"] = Dict{String,Any}("data" => D_test)

    # Get ground truth values for all train/validation/test sets
    @info "Solving train/validation/test sets"
    for (s, dataset) in [("train", D_train), ("val", D_val), ("test", D_test)]
        Y, Z, L, T = solve_dataset(dataset)
        D[s]["gurobi"] = Dict{String,Any}(
            "y" => Y,
            "obj_lp" => Z,
            "obj_lagrange" => L,
            "solve_time" => T,
        )
    end

    # Initialize DNN model
    @info "Initializing the DNN"
    dnn = gpu(MDK.initialize_fcnn(m, n; dc3))

    # Train the DNN model
    # For MD-knapsack problems, models tend to train pretty fast,
    # so only about 1000 epochs are needed
    @info "Training..."
    t0 = time()
    best_state = MDK.train!(dnn, D_train, D_val;
        minibatch_size=minibatch_size,
        η0=η0,
        ηmin=ηmin,
        max_epoch=max_epoch,
        max_epoch_no_progress=max_epoch_no_progress,
        freq_log=freq_log,
        # DC3 parameters
        dc3=dc3,
        dc3_corr_maxstep=dc3_corr_maxstep,
        dc3_corr_stepsize=dc3_corr_stepsize,
        dc3_corr_tol=dc3_corr_tol,
        dc3_loss_rho=dc3_loss_rho,
    )
    t_train = time() - t0
    @info "Training time: $(t_train)"

    # Recover DNN with best validation loss
    # Move everything to CPU to make ML vs GRB comparison easier
    dnn_best = MDK.initialize_fcnn(m, n; dc3) |> cpu
    Flux.loadmodel!(dnn_best, cpu(best_state))
    D["best_ML_state"] = cpu(best_state)

    @info "Computing train/val/test gaps"
    # TODO: benchmark model evaluation time
    for (s, dataset) in [("train", D_train), ("val", D_val), ("test", D_test)]
        Y, L = evaluate_dataset(dnn_best, dataset)

        Lgrb = D[s]["gurobi"]["obj_lagrange"]
        gap = (L - Lgrb) ./ Lgrb
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
            "training_time" => t_train,
        )
    end

    # Benchmark performance on test set
    if !dummy
        b = profile_dnn_inference(dnn_best, D_test; dc3, dc3_corr_maxstep, dc3_corr_stepsize)
        D["test"]["ml"]["inference_time"] = median(b.times ./ 1e6)
    end

    return D
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    m = parse(Int, ARGS[1])
    n = parse(Int, ARGS[2])
    s = parse(Int, ARGS[3])
    dc3 = parse(Bool, ARGS[4])
    resdir = ARGS[5]
    if !isdir(resdir)
        println("Invalid result directory: $(resdir)")
        exit(1)
    end

    method = dc3 ? "DC3" : "DLL"

    @info "Compilation run"
    main(2, 5; dummy=true, N_train=128, N_val=64, N_test=64, max_epoch=64, freq_log=0, dc3=dc3)

    @info "Actual run (method=$method)"
    D = main(m, n; seed=s, 
        # Learning settings
        max_epoch=1024,
        max_epoch_no_progress=32,
        η0=1e-4,
        ηmin=1e-7,
        # DC3 parameters
        dc3=dc3,
        dc3_corr_maxstep=10,
        dc3_corr_stepsize=1f-4,
        dc3_loss_rho=10f0,
    )
    println("ML results ($(method)):")
    print_result_table(D)  # for user convenience
    println()
    JLD2.save(joinpath(resdir, "mdk_m$(m)_n$(n)_s$(s)_$(method).jld2"), D)

    exit(0)
end
