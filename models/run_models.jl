
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

include("$(@__DIR__)/parameters.jl")
include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/read_demand.jl")

RCoeffs = RackPlacementCoefficients()

DC = build_datacenter("$(@__DIR__)/../data/contiguousDataCenterNew")
Sim = HistoricalDemandSimulator("$(@__DIR__)/../data/syntheticDemandSimulation")
batches, batch_sizes = read_demand("$(@__DIR__)/../data/demandTrajectories/150res_1.csv", RCoeffs)


oracle_result = rack_placement_oracle(batches, batch_sizes, DC)
println(oracle_result["x"])
println(oracle_result["y"])
println(oracle_result["objective"])
println(oracle_result["time_taken"])

SSOA_result = rack_placement(DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SSOA", S = 1, seed = 0)
SAA_result = rack_placement(DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SAA", S = 5, seed = 0)
MPC_result = rack_placement(DC, Sim, RCoeffs, batches, batch_sizes, strategy = "MPC")

SSOA_result["objective"]
SAA_result["objective"]
MPC_result["objective"]




SSOA_result["time_taken"][end]
SAA_result["time_taken"][end]
MPC_result["time_taken"][end]


