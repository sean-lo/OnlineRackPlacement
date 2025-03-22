include("$(@__DIR__)/read_demand.jl")
include("$(@__DIR__)/simulate_batch.jl")
using CSV
using DataFrames
using Glob
using Random

Sim = HistoricalDemandSimulator("$(@__DIR__)/../data/syntheticDemandSimulation")
batches, batch_sizes = read_demand("$(@__DIR__)/../data/demandTrajectories/150res_1.csv", 200.0, 0.0, false)
@time begin
    simulate_batches_all(
        "SSOA", Sim, 200.0, 0.0, batch_sizes;
        S = 5,
        seed = 0,
    );
    nothing
end
