include("$(@__DIR__)/utils.jl")
include("$(@__DIR__)/parameters.jl")

function _qcdf_interpolated(
    quantile::Float64,
    quantiles::Vector{Float64},
    values::Vector{T},
) where {T <: Number}
    right_ind = findfirst(quantile .<= quantiles)
    if right_ind == 1
        return values[right_ind]
    end
    if isnothing(right_ind)
        error("ArgumentError: quantile = $quantile not found in quantiles = $quantiles")
    end
    qA, vA = quantiles[right_ind - 1], values[right_ind - 1]
    qB, vB = quantiles[right_ind], values[right_ind]
    return vA + (vB - vA) * (quantile - qA) / (qB - qA)
end

function _qcdf_interpolated(
    quantile_vals::Array{Float64},
    quantiles::Vector{Float64},
    values::Vector{Float64},
)
    right_inds = [
        findfirst(quantile .<= quantiles)
        for quantile in quantile_vals
    ]
    if any(isnothing.(right_inds))
        error("ArgumentError: quantile = $quantile_vals not found in quantiles = $quantiles")
    end
    if any(right_inds .== 1)
        error()
    end
    qA, vA = quantiles[right_inds .- 1], values[right_inds .- 1]
    qB, vB = quantiles[right_inds], values[right_inds]
    return vA .+ (vB .- vA) .* (quantile_vals .- qA) ./ (qB .- qA)
end

function _qcdf(
    quantile::Float64,
    quantiles::Vector{Float64},
    values::Vector{T},
) where {T <: Number}
    right_ind = findfirst(quantile .<= quantiles)
    return values[right_ind]
end

function _qcdf(
    quantile_vals::Array{Float64},
    quantiles::Vector{Float64},
    values::Vector{T},
) where {T <: Number}
    right_inds = [
        findfirst(quantile .<= quantiles)
        for quantile in quantile_vals
    ]
    return values[right_inds]
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
    size_quantiles = cumsum(distr_data["size"][!, "frequency"])
    size_quantiles = round.(size_quantiles, digits = 4)
    size_quantiles[end] = 1.0
    size_mean = round(sum(distr_data["size"][!, "size"] .* distr_data["size"][!, "frequency"]))

    sort!(distr_data["power"], [:powerPerDemandItem])
    power_quantiles = cumsum(distr_data["power"][!, "frequency"])
    power_quantiles = round.(power_quantiles, digits = 9)
    power_quantiles[end] = 1.0
    power_mean = round(sum(distr_data["power"][!, "powerPerDemandItem"] .* distr_data["power"][!, "frequency"]))

    sort!(distr_data["cooling"], [:coolingPerDemandItem])
    cooling_quantiles = cumsum(distr_data["cooling"][!, "frequency"])
    cooling_quantiles = round.(cooling_quantiles, digits = 6)
    cooling_quantiles[end] = 1.0
    cooling_mean = round(sum(distr_data["cooling"][!, "coolingPerDemandItem"] .* distr_data["cooling"][!, "frequency"]))
    return HistoricalDemandSimulator(
        distr_data["size"][!, "size"],
        size_mean,
        size_quantiles,
        distr_data["power"][!, "powerPerDemandItem"],
        power_mean,
        power_quantiles,
        distr_data["cooling"][!, "coolingPerDemandItem"],
        cooling_mean,
        cooling_quantiles,
    )
end

function simulate_demand(
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    batch_size::Int = 1,
    seed::Union{Int, Nothing} = nothing,
)
    if isnothing(seed)
        seed = abs(Random.rand(Int))
    end
    Random.seed!(seed)
    size_vals = [
        _qcdf(rand(), Sim.size_quantiles, Sim.size_endpoints)
        for _ in 1:batch_size
    ]
    power_vals = [
        _qcdf_interpolated(rand(), Sim.power_quantiles, Sim.power_endpoints)
        for _ in 1:batch_size
    ]
    cooling_vals = [
        _qcdf_interpolated(rand(), Sim.cooling_quantiles, Sim.cooling_endpoints)
        for _ in 1:batch_size
    ]
    return Dict(
        "seed" => seed,
        "size" => size_vals,
        "power" => power_vals,
        "cooling" => cooling_vals,
        "reward" => (placement_reward .+ placement_var_reward .* size_vals),
    )
end

function mean_demand(
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    batch_size::Int,
)
    return Dict(
        "seed" => 0,
        "size" => [Sim.size_mean for _ in 1:batch_size],
        "power" => [Sim.power_mean for _ in 1:batch_size],
        "cooling" => [Sim.cooling_mean for _ in 1:batch_size],
        "reward" => fill(placement_reward + Sim.size_mean * placement_var_reward, batch_size),
    )
end


function simulate_demands(
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    shape::Tuple{Vararg{Int}},
    seed::Union{Int, Nothing} = nothing,
)    
    if isnothing(seed)
        seed = abs(Random.rand(Int))
    end
    Random.seed!(seed)
    size_vals = _qcdf(rand(Float64, shape), Sim.size_quantiles, Sim.size_endpoints)
    power_vals = _qcdf_interpolated(rand(Float64, shape), Sim.power_quantiles, Sim.power_endpoints)
    cooling_vals = _qcdf_interpolated(rand(Float64, shape), Sim.cooling_quantiles, Sim.cooling_endpoints)
    return convert(Dict{String, <:Array}, Dict(
        "size" => size_vals,
        "power" => power_vals,
        "cooling" => cooling_vals,
        "reward" => placement_reward .+ placement_var_reward .* size_vals,
    ))
end

function mean_demands(
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    shape::Tuple{Vararg{Int}},
)
    return convert(Dict{String, <:Array}, Dict(
        "size" => fill(Sim.size_mean, shape),
        "power" => fill(Sim.power_mean, shape),
        "cooling" => fill(Sim.cooling_mean, shape),
        "reward" => fill(placement_reward + Sim.size_mean * placement_var_reward, shape),
    ))
end


function simulate_batches(
    strategy::String,
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    t::Int,
    T::Int,
    batch_sizes::Vector{Int},
    ;
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
        sim_batches = [
            simulate_demands(
                Sim, placement_reward, placement_var_reward,
                (maximum(batch_sizes[t+1:T]),),
            )
            for _ in 1:T-t
        ]
    elseif strategy == "SAA"
        sim_batches = [
            simulate_demands(
                Sim, placement_reward, placement_var_reward,
                (S, maximum(batch_sizes[t+1:T]),),
            )
            for _ in 1:T-t
        ]
    elseif strategy == "MPC"
        sim_batches = [
            mean_demands(
                Sim, placement_reward, placement_var_reward,
                (maximum(batch_sizes[t+1:T]),),
            )
            for _ in 1:T-t
        ]
    end
    return sim_batches
end

function simulate_batches_all(
    strategy::String,
    Sim::HistoricalDemandSimulator,
    placement_reward::Float64,
    placement_var_reward::Float64,
    batch_sizes::Vector{Int},
    ;
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
    T = length(batch_sizes)
    
    all_sim_batches = Vector{<:Dict{String, <:Array}}[]
    all_seeds = Random.rand(Int, T)
    for t in 1:T-1
        push!(all_sim_batches, simulate_batches(
            strategy, Sim, placement_reward, placement_var_reward,
            t, T, batch_sizes,
            ;
            S = S,
            seed = all_seeds[t],
        ))
    end
    return all_sim_batches
end