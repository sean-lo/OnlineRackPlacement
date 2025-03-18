
using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using DataFrames
using JuMP
using Gurobi
using StatsBase

include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")


DC = build_datacenter("$(@__DIR__)/../data/contiguousDataCenterNew")
Sim = HistoricalDemandSimulator("$(@__DIR__)/../data/syntheticDemandSimulation")

demand_dir = "$(@__DIR__)/../data/demandTrajectories"
demand_data = CSV.read(joinpath(demand_dir, "150res_1.csv"), DataFrame)
sort!(demand_data, :resID)


CONST_BATCH_SIZE = 10
T = Int(ceil(nrow(demand_data) / CONST_BATCH_SIZE))
batches = Dict(
    t => Dict(
        "seed" => 0,
        "size" => demand_data[inds, :size],
        "cooling" => demand_data[inds, :coolingEach],
        "power" => demand_data[inds, :powerEach],
        "reward" => ones(length(demand_data[inds, :resID])),
    )
    for (t, inds) in enumerate(Iterators.partition(1:nrow(demand_data), CONST_BATCH_SIZE))
)
batch_sizes = Dict(
    t => CONST_BATCH_SIZE
    for t in 1:T
)


oracle_result = build_solve_oracle_model(batches, batch_sizes, DC)
println(oracle_result["x"])
println(oracle_result["y"])
println(oracle_result["objective"])
println(oracle_result["time_taken"])

SSOA_result = rack_placement(DC, Sim, batches, batch_sizes, strategy = "SSOA", S = 1, seed = 0)
SAA_result = rack_placement(DC, Sim, batches, batch_sizes, strategy = "SAA", S = 5, seed = 0)
MPC_result = rack_placement(DC, Sim, batches, batch_sizes, strategy = "MPC")

SSOA_result["objective"]
SAA_result["objective"]
MPC_result["objective"]




SSOA_result["time_taken"][end]
SAA_result["time_taken"][end]
MPC_result["time_taken"][end]


