
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
include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/read_demand.jl")


function run_experiment(
    datacenter_dir::String,
    distr_dir::String,
    demand_fp::String,
    result_dir::String,
    ;
    write::Bool = true,
    env::Union{Gurobi.Env, Nothing} = nothing,
    strategy::String = "SSOA",
    discount_factor::Float64 = 0.1,
    use_batching::Bool = false,
    batch_size::Int = CONST_BATCH_SIZE,
    with_precedence::Bool = false,
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
    placement_reward::Float64 = 200.0,
    placement_var_reward::Float64 = 0.0,
    room_mult::Float64 = 1.0,
    row_mult::Float64 = 1.0,
    tilegroup_penalty::Float64 = 1.0,
    power_surplus_penalty::Float64 = 1e-3,
    power_balance_penalty::Float64 = 1e-5,
    S::Int = 1,
    seed::Int = 0,
    MIPGap::Float64 = 1e-4,
    time_limit_sec_per_iteration = 300,
    test_run::Bool = false,
    verbose::Bool = false,
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

    DC = build_datacenter(datacenter_dir)
    Sim = HistoricalDemandSimulator(distr_dir)
    batches, batch_sizes = read_demand(
        demand_fp, 
        RCoeffs.placement_reward, RCoeffs.placement_var_reward, 
        use_batching, batch_size,
    )

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
        r = postprocess_results_oracle(DC, result, batches)
    elseif strategy == "myopic"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "myopic",
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
        r = postprocess_results(
            result["all_results"], batch_sizes, DC, "myopic";
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
        )
    elseif strategy == "MPC"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "MPC",
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
        r = postprocess_results(
            result["all_results"], batch_sizes, DC, "MPC";
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
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
        r = postprocess_results(
            result["all_results"], batch_sizes, DC, "SSOA";
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
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
        r = postprocess_results(
            result["all_results"], batch_sizes, DC, "SAA";
            with_precedence = with_precedence,
            obj_minimize_rooms = obj_minimize_rooms,
            obj_minimize_rows = obj_minimize_rows,
            obj_minimize_tilegroups = obj_minimize_tilegroups,
            obj_minimize_power_surplus = obj_minimize_power_surplus,
            obj_minimize_power_balance = obj_minimize_power_balance,
        )
    end

    if write && !test_run
        mkpath(result_dir)
        if strategy != "oracle"
            CSV.write("$(result_dir)/iteration_data.csv", r["iteration_data"])
            CSV.write("$(result_dir)/room_space_utilization.csv", r["room_space_utilization_data"])
            CSV.write("$(result_dir)/toppower_utilization.csv", r["toppower_utilization_data"])
            if obj_minimize_power_surplus || obj_minimize_power_balance
                CSV.write("$(result_dir)/toppower_pair_utilization.csv", r["toppower_pair_utilization_data"])
            end
        else
            CSV.write("$(result_dir)/objective.csv", DataFrame(Dict("objective" => [result["objective"]])))
        end
    end

    if strategy != "oracle"
        optimality_gap_vals = r["iteration_data"][!, "optimality_gap"]
        optimality_gap_max = maximum(optimality_gap_vals)
        optimality_gap_mean = StatsBase.geomean(optimality_gap_vals .+ 1.0) - 1.0
        returnval = Dict(
            "time_taken" => result["time_taken"],
            "optimality_gap_max" => optimality_gap_max,
            "optimality_gap_mean" => optimality_gap_mean,
            "demands_placed" => r["demands_placed"],
            "racks_placed" => r["racks_placed"],
            "objective" => result["objective"],
            "toppower_utilization" => r["toppower_utilization"],
        )
        if with_precedence
            # Values until (and including) the first drop
            returnval["demands_placed_precedence"] = r["demands_placed_precedence"]
            returnval["racks_placed_precedence"] = r["racks_placed_precedence"]
            returnval["objective_precedence"] = r["objective_precedence"]
        end
    else
        returnval = Dict(
            "time_taken" => result["time_taken"],
            "optimality_gap_max" => result["optimality_gap"],
            "optimality_gap_mean" => result["optimality_gap"],
            "demands_placed" => result["demands_placed"],
            "racks_placed" => result["racks_placed"],
            "objective" => result["objective"],
            "toppower_utilization" => r["toppower_utilization"],
        )
    end

    return returnval
end