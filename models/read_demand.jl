const CONST_BATCH_SIZE = 10
include("$(@__DIR__)/parameters.jl")

function read_demand(
    demand_fp::String,
    placement_reward::Float64,
    placement_var_reward::Float64 = 0.0,
    use_batching::Bool = false,
    batch_size::Int = CONST_BATCH_SIZE,
)
    demand_data = CSV.read(demand_fp, DataFrame)
    if use_batching && (:batchID in names(demand_data))
        T = maximum(demand_data[!, :batchID])
        batches = Dict{String, <:Array}[]
        for t in 1:T
            sizes = demand_data[demand_data[!, :batchID] .== t, :size]
            push!(batches, Dict(
                "size" => sizes,
                "cooling" => demand_data[demand_data[!, :batchID] .== t, :coolingEach],
                "power" => demand_data[demand_data[!, :batchID] .== t, :powerEach],
                "reward" => (placement_reward .+ placement_var_reward .* sizes),
            ))
        end
    else
        T = Int(ceil(nrow(demand_data) / batch_size))
        batches = Dict{String, <:Array}[]
        for inds in Iterators.partition(1:nrow(demand_data), batch_size)
            sizes = demand_data[inds, :size]
            push!(batches, Dict(
                "size" => sizes,
                "cooling" => demand_data[inds, :coolingEach],
                "power" => demand_data[inds, :powerEach],
                "reward" => (placement_reward .+ placement_var_reward .* sizes),
            ))
        end
    end
    batch_sizes = [
        length(batches[t]["size"])
        for t in 1:T
    ]
    return batches, batch_sizes
end