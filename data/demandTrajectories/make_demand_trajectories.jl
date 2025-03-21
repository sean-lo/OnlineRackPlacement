using Pkg
Pkg.activate("$(@__DIR__)/../..")

include("$(@__DIR__)/../../models/simulate_batch.jl")

Sim = HistoricalDemandSimulator("$(@__DIR__)/../../data/syntheticDemandSimulation/")
RCoeffs = RackPlacementCoefficients()

for run_ind in 1:5
    demand = simulate_demand(Sim, RCoeffs.placement_reward, 150, run_ind)
    demand_df = DataFrame(
        coolingEach = demand["cooling"],
        powerEach = demand["power"],
        size = demand["size"],
        resID = 1:150,
    )
    CSV.write("$(@__DIR__)/150res_$run_ind.csv", demand_df)
end
