
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
    strategy::String = "SSOA",
    discount_factor::Float64 = 0.1,
    use_batching::Bool = false,
    batch_size::Int = CONST_BATCH_SIZE,
    online_objectives::Bool = true,
    S::Int = 1,
    seed::Int = 0,
    time_limit_sec_per_iteration = 300,
)
    RCoeffs = RackPlacementCoefficients(
        discount_factor = discount_factor,
    )

    DC = build_datacenter(datacenter_dir)
    Sim = HistoricalDemandSimulator(distr_dir)
    batches, batch_sizes = read_demand(demand_fp, RCoeffs, use_batching, batch_size)
    
    if strategy == "oracle"
        oracle_result = rack_placement_oracle(batches, batch_sizes, DC)
    elseif strategy == "myopic"
        myopic_result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, strategy = "myopic",
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
        )
        r = postprocess_results(myopic_result["all_results"], "myopic")
    elseif strategy == "MPC"
        MPC_result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, strategy = "MPC",
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
        r = postprocess_results(MPC_result["all_results"], "MPC")
    elseif strategy == "SSOA"
        SSOA_result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SSOA", S = 1, seed = seed,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
        r = postprocess_results(SSOA_result["all_results"], "SSOA")
    elseif strategy == "SAA"
        SAA_result = rack_placement(
            DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SAA", S = S, seed = seed,
            time_limit_sec_per_iteration = time_limit_sec_per_iteration,
            obj_minimize_rooms = online_objectives,
            obj_minimize_rows = online_objectives,
            obj_minimize_tilegroups = online_objectives,
            obj_minimize_power_surplus = online_objectives,
            obj_minimize_power_balance = online_objectives,
        )
        r = postprocess_results(SAA_result["all_results"], "SAA")
    end

    if strategy != "oracle"
        CSV.write("$(result_dir)/$(strategy)_$(seed).csv", r["iteration_data"])
        CSV.write("$(result_dir)/$(strategy)_$(seed)_room_space_utilization.csv", r["room_space_utilization_data"])
        if online_objectives
            CSV.write("$(result_dir)/$(strategy)_$(seed)_toppower_utilization.csv", r["toppower_utilization_data"])
            CSV.write("$(result_dir)/$(strategy)_$(seed)_toppower_pair_utilization.csv", r["toppower_pair_utilization_data"])
        end
    end
end


### Filepaths ### 
fps = JSON.parsefile("$(@__DIR__)/../filepaths.json")
datacenter_dir = fps["datacenter_dir"]
distr_dir = fps["distr_dir"]
demand_fp = fps["demand_fp"]
result_dir = "$(@__DIR__)/../experiments/online_0.1"
### 

run_experiment(
    datacenter_dir,
    distr_dir,
    demand_fp,
    result_dir,
    ;
    strategy = "SSOA",
    time_limit_sec_per_iteration = 20,
    online_objectives = true,
)