
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
    online_objectives::Bool = true,
    S::Int = 1,
    seed::Int = 0,
    MIPGap::Float64 = 1e-4,
    time_limit_sec_per_iteration = 300,
)
    if isnothing(env)
        env = Gurobi.Env()
    end
    RCoeffs = RackPlacementCoefficients(
        discount_factor = discount_factor,
    )

    DC = build_datacenter(datacenter_dir)
    Sim = HistoricalDemandSimulator(distr_dir)
    batches, batch_sizes = read_demand(demand_fp, RCoeffs, use_batching, batch_size)
    
    if strategy == "oracle"
        r = postprocess_results_oracle(DC, result, batches, batch_sizes)
    elseif strategy == "myopic"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "myopic",
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
            MIPGap = MIPGap,
        )
        r = postprocess_results(
            result["all_results"], DC, "myopic";
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
    elseif strategy == "MPC"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "MPC",
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
            MIPGap = MIPGap,
        )
        r = postprocess_results(
            result["all_results"], DC, "MPC";
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
    elseif strategy == "SSOA"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "SSOA", S = 1, seed = seed,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
            MIPGap = MIPGap,
        )
        r = postprocess_results(
            result["all_results"], DC, "SSOA";
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
    elseif strategy == "SAA"
        result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, 
            env = env,
            strategy = "SAA", S = S, seed = seed,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
            MIPGap = MIPGap,
        )
        r = postprocess_results(
            result["all_results"], DC, "SAA";
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
    end

    if write 
        if strategy != "oracle"
            CSV.write("$(result_dir)/iteration_data.csv", r["iteration_data"])
            CSV.write("$(result_dir)/room_space_utilization.csv", r["room_space_utilization_data"])
            if online_objectives
                CSV.write("$(result_dir)/toppower_utilization.csv", r["toppower_utilization_data"])
                CSV.write("$(result_dir)/toppower_pair_utilization.csv", r["toppower_pair_utilization_data"])
            end
        else
            CSV.write("$(result_dir)/objective.csv", DataFrame(Dict("objective" => [result["objective"]])))
        end
    end

    if strategy != "oracle"
        optimality_gap_vals = r["iteration_data"][!, "optimality_gap"]
        optimality_gap_mean = StatsBase.geomean(optimality_gap_vals .+ 1.0) - 1.0
        returnval = Dict(
            "time_taken" => result["time_taken"],
            "optimality_gap_mean" => optimality_gap_mean,
            "demands_placed" => Int(round((r["iteration_data"][!, "current_assignment"] |> sum) / (RCoeffs.placement_reward))),
        )
    else
        optimality_gap_mean = 0.0
        returnval = Dict(
            "time_taken" => result["time_taken"],
            "optimality_gap_mean" => optimality_gap_mean,
            "demands_placed" => Int(round(result["objective"] / (RCoeffs.placement_reward))),
        )
    end

    return returnval
end