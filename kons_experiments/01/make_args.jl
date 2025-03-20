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

discount_factor_range = [0.1, 1.0]
online_objectives_range = [false, true]

args_df = DataFrame(
    run_ind = Int[],
    datacenter_dir = String[],
    distr_dir = String[],
    demand_fp = String[],
    method = String[],
    # Index of method
    nummethod = Int[],
    online_objectives = Bool[],
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
                false, 0.0, 
                1, 1,
            ))
        elseif method == "myopic"
            for online_objectives in online_objectives_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    online_objectives, 0.0, 
                    1, 1,
                ))
            end
        elseif method == "MPC"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    online_objectives, discount_factor, 
                    1, 1,
                ))
            end
        elseif method == "SSOA"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range,
                seed in seed_range
                push!(args_df, (
                    run_ind, 
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    online_objectives, discount_factor, 
                    1, seed,
                ))
            end
        elseif method == "SAA"
            for discount_factor in discount_factor_range,
                online_objectives in online_objectives_range,
                seed in seed_range,
                S in S_list
                push!(args_df, (
                    run_ind,     
                    datacenter_dir, distr_dir, demand_fp,
                    method, nummethod,
                    online_objectives, discount_factor, 
                    S, seed,
                ))
            end
        end
    end
end

sort!(args_df, [:nummethod, :online_objectives, :discount_factor, :S, :seed])
CSV.write("$(@__DIR__)/args.csv", args_df)

test_args_df = filter(
    r -> r.method == "oracle" || (
        r.run_ind == 1
        && r.online_objectives == true
        && r.seed == 1
    ),
    args_df
)
unique!(test_args_df, [:method])
CSV.write("$(@__DIR__)/test_args.csv", test_args_df)

