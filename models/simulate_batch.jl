include("$(@__DIR__)/utils.jl")

using CSV
using DataFrames
using StatsBase
using Random

function _qcdf_interpolated(
    quantiles::Vector{Float64},
    values::Vector{T},
    quantile::Float64,
) where {T <: Number}
    right_ind = findfirst(quantile .<= quantiles)
    if right_ind == 1
        return values[right_ind]
    end
    qA, vA = quantiles[right_ind - 1], values[right_ind - 1]
    qB, vB = quantiles[right_ind], values[right_ind]
    return vA + (vB - vA) * (quantile - qA) / (qB - qA)
end

function _qcdf(
    quantiles::Vector{Float64},
    values::Vector{T},
    quantile::Float64,
) where {T <: Number}
    right_ind = findfirst(quantile .<= quantiles)
    return values[right_ind]
end

struct HistoricalDemandSimulator
    size_endpoints::Vector{Int}
    size_mean::Float64
    size_quantiles::Vector{Float64}
    power_endpoints::Vector{Float64}
    power_mean::Float64
    power_quantiles::Vector{Float64}
    cooling_endpoints::Vector{Float64}
    cooling_mean::Float64
    cooling_quantiles::Vector{Float64}
end

function HistoricalDemandSimulator(
    data_dir::String,
)
    distr_data = read_CSVs_from_dir(data_dir)

    sort!(distr_data["size"], [:size])
    distr_data["size"][!, "cumFrequency"] = cumsum(distr_data["size"][!, "frequency"])
    size_mean = round(sum(distr_data["size"][!, "size"] .* distr_data["size"][!, "frequency"]))

    sort!(distr_data["power"], [:powerPerDemandItem])
    distr_data["power"][!, "cumFrequency"] = cumsum(distr_data["power"][!, "frequency"])
    power_mean = round(sum(distr_data["power"][!, "powerPerDemandItem"] .* distr_data["power"][!, "frequency"]))

    sort!(distr_data["cooling"], [:coolingPerDemandItem])
    distr_data["cooling"][!, "cumFrequency"] = cumsum(distr_data["cooling"][!, "frequency"])
    cooling_mean = round(sum(distr_data["cooling"][!, "coolingPerDemandItem"] .* distr_data["cooling"][!, "frequency"]))
    return HistoricalDemandSimulator(
        distr_data["size"][!, "size"],
        size_mean,
        distr_data["size"][!, "cumFrequency"],
        distr_data["power"][!, "powerPerDemandItem"],
        power_mean,
        distr_data["power"][!, "cumFrequency"],
        distr_data["cooling"][!, "coolingPerDemandItem"],
        cooling_mean,
        distr_data["cooling"][!, "cumFrequency"],
    )
end

function simulate_demand(
    Sim::HistoricalDemandSimulator,
    batch_size::Int = 1,
    seed::Union{Int, Nothing} = nothing,
)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    size_vals = [
        _qcdf(Sim.size_quantiles, Sim.size_endpoints, rand())
        for _ in 1:batch_size
    ]
    power_vals = [
        _qcdf_interpolated(Sim.power_quantiles, Sim.power_endpoints, rand())
        for _ in 1:batch_size
    ]
    cooling_vals = [
        _qcdf_interpolated(Sim.cooling_quantiles, Sim.cooling_endpoints, rand())
        for _ in 1:batch_size
    ]
    return Dict(
        "seed" => seed,
        "size" => size_vals,
        "power" => power_vals,
        "cooling" => cooling_vals,
        "reward" => ones(batch_size),
    )
end

function mean_demand(
    Sim::HistoricalDemandSimulator,
    batch_size::Int,
)
    return Dict(
        "seed" => 0,
        "size" => [Sim.size_mean for _ in 1:batch_size],
        "power" => [Sim.power_mean for _ in 1:batch_size],
        "cooling" => [Sim.cooling_mean for _ in 1:batch_size],
        "reward" => ones(batch_size),
    )
end


function simulate_batches(
    strategy::String,
    Sim::HistoricalDemandSimulator,
    t::Int,
    T::Int,
    batch_sizes::Dict{Int, Int},
    S::Int = 1,
    seed::Union{Int, Nothing} = nothing,
)
    if isnothing(seed)
        seed = abs(Random.rand(Int))
    end
    if !(strategy in ["SSOA", "SAA", "MPC"])
        error("ArgumentError: strategy = $strategy not recognized.")
    end
    Random.seed!(seed)
    if strategy == "SSOA"
        sim_batches = Dict(
            τ => simulate_demand(Sim, batch_sizes[τ])
            for τ in t+1:T
        )
        sim_batch_sizes = Dict(
            τ => length(sim_batches[τ]["size"])
            for τ in t+1:T
        )
    elseif strategy == "SAA"
        sim_batches = Dict(
            (τ, s) => simulate_demand(Sim, batch_sizes[τ])
            for τ in t+1:T, s in 1:S
        )
        sim_batch_sizes = Dict(
            (τ, s) => length(sim_batches[(τ, s)]["size"])
            for τ in t+1:T, s in 1:S
        )
    elseif strategy == "MPC"
        sim_batches = Dict(
            τ => mean_demand(Sim, batch_sizes[τ])
            for τ in t+1:T
        )
        sim_batch_sizes = Dict(
            τ => length(sim_batches[τ]["size"])
            for τ in t+1:T
        )
    end
    return sim_batches, sim_batch_sizes
end