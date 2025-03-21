using Random
using CSV, DataFrames
using JSON

methodlist = [
    "oracle", 
    "myopic", 
    "MPC", 
    "SSOA", 
    "SAA",
]
S_list = [5]
seed_range = collect(1:5)
fps = JSON.parsefile("$(@__DIR__)/../filepaths.json")
datacenter_dir = fps["datacenter_dir"]
distr_dir = fps["distr_dir"]
demand_fps = fps["demand_fps"]

discount_factor_range = [0.1, 0.5, 1.0]
use_batching_range = [false]
batch_size_range = [10]
online_objectives_range = [
    # obj_minimize_rooms
    # obj_minimize_rows
    # obj_minimize_tilegroups
    # obj_minimize_power_surplus
    # obj_minimize_power_balance
    (false, false, false, false, false),
    (false, true, false, true, true),
]
online_params = [
    # room_mult,
    # row_mult,
    # tilegroup_penalty,
    # power_surplus_penalty,
    # power_balance_penalty,
    (0.0, 1.0, 0.0, 1e-3, 1e-5)
]

args_df = DataFrame(
    run_ind = Int[],
    datacenter_dir = String[],
    distr_dir = String[],
    demand_fp = String[],
    method = String[],
    # Index of method
    nummethod = Int[],
    use_batching = Bool[],
    batch_size = Int[],
    obj_minimize_rooms = Bool[],
    obj_minimize_rows = Bool[],
    obj_minimize_tilegroups = Bool[],
    obj_minimize_power_surplus = Bool[],
    obj_minimize_power_balance = Bool[],
    room_mult = Float64[],
    row_mult = Float64[],
    tilegroup_penalty = Float64[],
    power_surplus_penalty = Float64[],
    power_balance_penalty = Float64[],
    discount_factor = Float64[],
    # Number of sample paths (for SAA)
    S = Int[],
    # seed for random inits of realizations 
    seed = Int[], 
)

for (run_ind, demand_fp) in enumerate(demand_fps)
    for (nummethod, method) in enumerate(methodlist)
        if method == "oracle"
            push!(args_df, (
                run_ind, 
                datacenter_dir, distr_dir, demand_fp,
                method, nummethod,
                false, batch_size_range[1],
                online_objectives_range[1]...,
                online_params[1]...,
                0.0, 
                1, 1,
            ))
        elseif method == "myopic"
            for online_objectives in online_objectives_range,
                use_batching in use_batching_range,
                batch_size in batch_size_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    use_batching, batch_size,
                    online_objectives...,
                    online_params[1]..., 0.0, 
                    1, 1,
                ))
            end
        elseif method == "MPC"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range,
                use_batching in use_batching_range,
                batch_size in batch_size_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    use_batching, batch_size,
                    online_objectives...,
                    online_params[1]..., discount_factor, 
                    1, 1,
                ))
            end
        elseif method == "SSOA"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range,
                use_batching in use_batching_range,
                batch_size in batch_size_range,
                seed in seed_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    use_batching, batch_size,
                    online_objectives...,
                    online_params[1]..., discount_factor, 
                    1, seed,
                ))
            end
        elseif method == "SAA"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range,
                use_batching in use_batching_range,
                batch_size in batch_size_range,
                seed in seed_range,
                S in S_list
                push!(args_df, (
                    run_ind,     
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    use_batching, batch_size,
                    online_objectives...,
                    online_params[1]..., discount_factor, 
                    S, seed,
                ))
            end
        end
    end
end

sort!(args_df, [:nummethod, :obj_minimize_rows, :discount_factor, :S, :seed])
CSV.write("$(@__DIR__)/args.csv", args_df)

test_args_df = filter(
    r -> r.method == "oracle" || (
        r.run_ind == 1
        && r.obj_minimize_rows == true
        && r.seed == 1
    ),
    args_df
)
unique!(test_args_df, [:method])
CSV.write("$(@__DIR__)/test_args.csv", test_args_df)

