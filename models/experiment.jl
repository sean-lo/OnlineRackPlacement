
using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using DataFrames
using JuMP
using Gurobi
using Parameters
using Random
using StatsBase
using JSON
using Combinatorics
using Printf


include("$(@__DIR__)/parameters.jl")
include("$(@__DIR__)/read_datacenter.jl")
include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/read_demand.jl")


function run_experiment(
    distr_dir::String,
    demand_fp::String,
    result_dir::String,
    ;
    datacenter_dir::Union{String, Nothing} = nothing,
    env::Union{Gurobi.Env, Nothing} = nothing,
    use_batching::Bool = false,
    batch_size::Int = CONST_BATCH_SIZE,
    lookahead_horizon::Union{Int, Nothing} = nothing,
    interpolate_power::Bool = true,
    interpolate_cooling::Bool = true,
    with_precedence::Bool = false,
    placement_reward::Float64 = 200.0,
    placement_var_reward::Float64 = 0.0,
    strategy::String = "SSOA",
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
    room_mult::Float64 = 1.0,
    row_mult::Float64 = 1.0,
    tilegroup_penalty::Float64 = 1.0,
    power_surplus_penalty::Float64 = 1e-3,
    power_balance_penalty::Float64 = 1e-5,
    discount_factor::Float64 = 0.1,
    S::Int = 1,
    seed::Int = 0,
    time_limit_sec_per_iteration = 300,
    MIPGap::Float64 = 1e-4,
    test_run::Bool = false,
    verbose::Bool = false,
    write::Bool = true,
    build_datacenter_kwargs...,
)
    if isnothing(env)
        env = Gurobi.Env()
    end
    verbose && println("Running experiment with strategy: $strategy")
    RCoeffs = RackPlacementCoefficients(
        placement_reward = placement_reward,
        placement_var_reward = placement_var_reward,
        room_mult = (obj_minimize_rooms ? room_mult : 0.0),
        row_mult = (obj_minimize_rows ? row_mult : 0.0),
        tilegroup_penalty = (obj_minimize_tilegroups ? tilegroup_penalty : 0.0),
        power_surplus_penalty = (obj_minimize_power_surplus ? power_surplus_penalty : 0.0),
        power_balance_penalty = (obj_minimize_power_balance ? power_balance_penalty : 0.0),
        discount_factor = discount_factor,
    )
    
    if isnothing(datacenter_dir) 
        DC = build_datacenter(
            ;
            build_datacenter_kwargs...,
        )
    else
        DC = read_datacenter(datacenter_dir)
    end
    Sim = HistoricalDemandSimulator(
        distr_dir,
        interpolate_power = interpolate_power,
        interpolate_cooling = interpolate_cooling,
    )
    batches, batch_sizes = read_demand(
        demand_fp, 
        RCoeffs.placement_reward, RCoeffs.placement_var_reward, 
        use_batching, batch_size,
    )
    if !isnothing(lookahead_horizon)
        num_future_periods = lookahead_horizon
    else
        num_future_periods = length(batches)
    end

    verbose && println("Created batches and simulator.")
    
    if strategy == "oracle"
        result = rack_placement_oracle(
            batches, batch_sizes, DC, 
            ;
            env = env,
            with_precedence = with_precedence,
            time_limit_sec = time_limit_sec_per_iteration,
            MIPGap = MIPGap,
        )
    elseif strategy == "myopic"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "myopic",
            num_future_periods = num_future_periods,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
            MIPGap = MIPGap,
            verbose = verbose,
            test_run = test_run,
        )
    elseif strategy == "MPC"
        all_sim_batches = simulate_batches_all(
            strategy, Sim, 
            RCoeffs.placement_reward, RCoeffs.placement_var_reward,
            batch_sizes, 
            ;
            S = 1,
            seed = seed,
            test_run = test_run,
        )
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "MPC",
            num_future_periods = num_future_periods,
            all_sim_batches = all_sim_batches,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
            MIPGap = MIPGap,
            verbose = verbose,
            test_run = test_run,
        )
    elseif strategy == "SSOA"
        verbose && println("Simulating batches for SSOA...")
        all_sim_batches = simulate_batches_all(
            strategy, Sim, 
            RCoeffs.placement_reward, RCoeffs.placement_var_reward,
            batch_sizes, 
            ;
            S = 1,
            seed = seed,
            test_run = test_run,
        )
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "SSOA", S = 1, 
            num_future_periods = num_future_periods,
            all_sim_batches = all_sim_batches,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
            MIPGap = MIPGap,
            verbose = verbose,
            test_run = test_run,
        )
    elseif strategy == "SAA"
        verbose && println("Simulating batches for SAA...")
        all_sim_batches = simulate_batches_all(
            strategy, Sim, 
            RCoeffs.placement_reward, RCoeffs.placement_var_reward,
            batch_sizes, 
            ;
            S = S,
            seed = seed,
            test_run = test_run,
        )
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "SAA", S = S, 
            num_future_periods = num_future_periods,
            all_sim_batches = all_sim_batches,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
            MIPGap = MIPGap,
            verbose = verbose,
            test_run = test_run,
        )
    end

    if write && !test_run
        mkpath(result_dir)
        open("$(result_dir)/x.json", "w") do f
            JSON.print(f, result["x"], 4)
        end
        open("$(result_dir)/y.json", "w") do f
            JSON.print(f, result["y"], 4)
        end
        open("$(result_dir)/u.json", "w") do f
            JSON.print(f, result["u"], 4)
        end
        if strategy != "oracle"
            CSV.write("$(result_dir)/iteration_data.csv", result["iteration_data"])
            CSV.write("$(result_dir)/row_space_utilization.csv", result["row_space_utilization_data"])
            CSV.write("$(result_dir)/room_space_utilization.csv", result["room_space_utilization_data"])
            CSV.write("$(result_dir)/toppower_utilization.csv", result["toppower_utilization_data"])
            if obj_minimize_power_surplus || obj_minimize_power_balance
                CSV.write("$(result_dir)/toppower_pair_utilization.csv", result["toppower_pair_utilization_data"])
            end
        else
            CSV.write("$(result_dir)/objective.csv", DataFrame(Dict("objective" => [result["objective"]])))
        end
    end

    return result
end