const CONST_BATCH_SIZE = 10

function read_demand(
    demand_fp::String,
    use_batching::Bool = false,
    batch_size::Int = CONST_BATCH_SIZE,
)
    demand_data = CSV.read(demand_fp, DataFrame)
    if use_batching && (:batchID in names(demand_data))
        T = maximum(demand_data[!, :batchID])
        batches = Dict{Int, Dict{String, Any}}()
        for t in 1:T
            batches[t] = Dict(
                "seed" => 0,
                "size" => demand_data[demand_data[!, :batchID] .== t, :size],
                "cooling" => demand_data[demand_data[!, :batchID] .== t, :coolingEach],
                "power" => demand_data[demand_data[!, :batchID] .== t, :powerEach],
                "reward" => ones(length(demand_data[demand_data[!, :batchID] .== t, :resID])),
            )
        end
    else
        T = Int(ceil(nrow(demand_data) / batch_size))
        batches = Dict(
            t => Dict(
                "seed" => 0,
                "size" => demand_data[inds, :size],
                "cooling" => demand_data[inds, :coolingEach],
                "power" => demand_data[inds, :powerEach],
                "reward" => ones(length(demand_data[inds, :resID])),
            )
            for (t, inds) in enumerate(Iterators.partition(1:nrow(demand_data), batch_size))
        )
    end
    batch_sizes = Dict(
        t => length(batches[t]["size"])
        for t in 1:T
    )
    return batches, batch_sizes
end