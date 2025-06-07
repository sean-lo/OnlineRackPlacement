using Pkg
Pkg.activate("$(@__DIR__)/../..")
using Glob
using DataFrames
using CSV
using Random
using Plots

include("$(@__DIR__)/../../models/simulate_batch.jl")

Sim = HistoricalDemandSimulator(
    "$(@__DIR__)/../../data/syntheticDemandSimulation/",
    interpolate_power = false,
    interpolate_cooling = false,
)
RCoeffs = RackPlacementCoefficients()

for run_ind in 1:20
    Random.seed!(run_ind)
    seed = abs.(Random.rand(Int))
    Random.seed!(seed)
    seed = abs.(Random.rand(Int))
    demand = simulate_demand(Sim, RCoeffs.placement_reward, RCoeffs.placement_var_reward, 150, seed)
    demand_df = DataFrame(
        coolingEach = demand["cooling"],
        powerEach = demand["power"],
        size = demand["size"],
        resID = 1:150,
    )
    CSV.write("$(@__DIR__)/new_150res_$run_ind.csv", demand_df)
end

size_df = CSV.read("$(@__DIR__)/../../data/syntheticDemandSimulation/size.csv", DataFrame)
Plots.bar(
    size_df[!, :size],
    size_df[!, :frequency],
    label = "Size",
    xlabel = "Size",
    ylabel = "Frequency",
    legend = :false,
)
for run_ind in 1:5
    demand_df = CSV.read("$(@__DIR__)/../../data/demandTrajectories/new_150res_$run_ind.csv", DataFrame)
    p = Plots.histogram(
        demand_df[!, :size], bins=0:1:21, normalize = :density, alpha = 0.5, label = "(New) run $run_ind",
        ylim = (0, 60),
    )
    Plots.display(p)
end

power_df = CSV.read("$(@__DIR__)/../../data/syntheticDemandSimulation/power.csv", DataFrame)
Plots.bar(
    power_df[!, :powerPerDemandItem],
    power_df[!, :frequency],
    label = "Power",
    xlabel = "Power",
    ylabel = "Frequency",
)




run_ind = 1
demand_df = CSV.read("$(@__DIR__)/../../data/demandTrajectories/new_150res_$run_ind.csv", DataFrame)
cooling_vals = demand_df[!, :coolingEach] .* demand_df[!, :size]

Sim = HistoricalDemandSimulator(
    "$(@__DIR__)/../../data/syntheticDemandSimulation/",
    interpolate_power = false,
    interpolate_cooling = false,
)
RCoeffs = RackPlacementCoefficients()

demand = simulate_demand(Sim, RCoeffs.placement_reward, RCoeffs.placement_var_reward, 150, 50)
demand_df = DataFrame(
    coolingEach = demand["cooling"],
    powerEach = demand["power"],
    size = demand["size"],
    resID = 1:150,
)
cooling_vals = demand_df[!, :coolingEach] .* demand_df[!, :size]
DC = read_datacenter("$(@__DIR__)/../../data/contiguousDataCenterNew/")

Sim.cooling_mean * 40