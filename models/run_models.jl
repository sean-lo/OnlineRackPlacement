
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
using JSON
using Combinatorics
using Printf

include("$(@__DIR__)/parameters.jl")
include("$(@__DIR__)/read_datacenter.jl")
include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/read_demand.jl")

include("$(@__DIR__)/experiment.jl")


### Filepaths ### 
# fps = JSON.parsefile("$(@__DIR__)/../filepaths.json")
fps = JSON.parsefile("$(@__DIR__)/../new_filepaths.json")
# fps = JSON.parsefile("$(@__DIR__)/new_filepaths.json")
datacenter_dir = fps["datacenter_dir"]
demand_fps = fps["demand_fps"]
run_ind = 1
distr_dir = fps["distr_dir"]
demand_fp = demand_fps[run_ind]
### 

# Parameters
use_batching = false
batch_size = 1
(
    obj_minimize_rooms, 
    obj_minimize_rows, 
    obj_minimize_tilegroups, 
    obj_minimize_power_surplus, 
    obj_minimize_power_balance,
) = (false, true, false, true, true)
# (
#     obj_minimize_rooms, 
#     obj_minimize_rows, 
#     obj_minimize_tilegroups, 
#     obj_minimize_power_surplus, 
#     obj_minimize_power_balance,
# ) = (false, false, false, false, false)
discount_factor = 1.0
with_precedence = true
interpolate_power = false
interpolate_cooling = false
placement_reward = 200.0
placement_var_reward = 0.0
room_mult = 0.0
row_mult = 1.0
tilegroup_penalty = 0.0
power_surplus_penalty = 1e-3
power_balance_penalty = 1e-5
MIPGap = 1e-2
time_limit_sec_per_iteration = 30
test_run = false
verbose = true
result_dir = "$(@__DIR__)/results/"


Sim = HistoricalDemandSimulator(
    distr_dir,
    interpolate_power = interpolate_power,
    interpolate_cooling = interpolate_cooling,
)

batches, batch_sizes = read_demand(
    demand_fp, placement_reward, placement_var_reward,
    use_batching, batch_size,
)
cooling_vals = vcat(
    [batches[t]["cooling"] .* batches[t]["size"] for t in 1:length(batches)]...
)
Plots.histogram(
    cooling_vals,
    bins = 0:1000:50000,
)

oracle_result = run_experiment(
    distr_dir,
    demand_fp,
    joinpath(result_dir, "oracle"),
    ;
    datacenter_dir = datacenter_dir,
    use_batching = use_batching,
    batch_size = batch_size,
    interpolate_power = interpolate_power,
    interpolate_cooling = interpolate_cooling,
    with_precedence = with_precedence,
    placement_reward = placement_reward,
    placement_var_reward = placement_var_reward,
    strategy = "oracle",
    obj_minimize_rooms = obj_minimize_rooms,
    obj_minimize_rows = obj_minimize_rows,
    obj_minimize_tilegroups = obj_minimize_tilegroups,
    obj_minimize_power_surplus = obj_minimize_power_surplus,
    obj_minimize_power_balance = obj_minimize_power_balance,
    room_mult = room_mult,
    row_mult = row_mult,
    tilegroup_penalty = tilegroup_penalty,
    power_surplus_penalty = power_surplus_penalty,
    power_balance_penalty = power_balance_penalty,
    discount_factor = discount_factor,
    S = 1,
    seed = 1,
    time_limit_sec_per_iteration = 60.0,
    MIPGap = MIPGap,
    num_threads = 2,
    test_run = test_run,
    verbose = verbose,
    write = true,
    cooling_capacity = 4e4,
)

DC = read_datacenter(datacenter_dir)
DC.power_capacity |> values |> unique

demand_fp
batches, batch_sizes = read_demand(demand_fp, placement_reward, placement_var_reward, use_batching, batch_size)
[x["power"][1] * x["size"][1] for x in batches] |> maximum
[k for k in keys(oracle_result["y"]) if k[1] == 1]
DC.row_tilegroups_map[12]
[k for k in keys(oracle_result["x"]) if k[1] == 1]
[
    p for p in DC.power_IDs if 28 in DC.power_tilegroups_map[p]
]
DC.power_tilegroups_map[77]
DC.power_tilegroups_map[92]
DC.power_capacity[77]
DC.power_capacity[92]
oracle_result["power_utilization"][77]
oracle_result["power_utilization"][92]


MPC_result = run_experiment(
    distr_dir,
    demand_fp,
    joinpath(result_dir, "MPC"),
    ;
    write = true,
    strategy = "MPC",
    discount_factor = discount_factor,
    with_precedence = with_precedence,
    obj_minimize_rooms = obj_minimize_rooms,
    obj_minimize_rows = obj_minimize_rows,
    obj_minimize_tilegroups = obj_minimize_tilegroups,
    obj_minimize_power_surplus = obj_minimize_power_surplus,
    obj_minimize_power_balance = obj_minimize_power_balance,
    interpolate_power = interpolate_power,
    interpolate_cooling = interpolate_cooling,
    placement_reward = placement_reward,
    placement_var_reward = placement_var_reward,
    room_mult = room_mult,
    row_mult = row_mult,
    tilegroup_penalty = tilegroup_penalty,
    power_surplus_penalty = power_surplus_penalty,
    power_balance_penalty = power_balance_penalty,
    S = 1,
    seed = 1,
    MIPGap = MIPGap,
    time_limit_sec_per_iteration = time_limit_sec_per_iteration,
    test_run = test_run,
    verbose = verbose,
    toppower_capacity = 9e5,
    midpower_capacity = 9e5 / 5,
    lowpower_capacity = 9e5 / (5 * 2),
    cooling_capacity = 40000.0,
)
MPC_result["objective"]
MPC_result["objective_precedence"]


# MPC_result["objective_precedence"]
MPC_result["power_utilization"]
MPC_result["cooling_utilization"]

Plots.histogram(
    MPC_result["cooling_utilization"] |> values |> collect,
    # bins=0:1:21
    bins = 0:2000:50000,
)

Plots.histogram(
    MPC_result["row_space_utilization_data"] |> Matrix |> vec,
    bins=0:1:21
)
MPC_result["room_space_utilization_data"]
MPC_result["toppower_utilization_data"]

using Plots
DC = build_datacenter()
Plots.histogram(
    [MPC_result["power_utilization"][p] for p in DC.lowpower_IDs],
    bins = 0:5000:120000,
)
Plots.histogram(
    [MPC_result["power_utilization"][p] for p in DC.midpower_IDs],
    bins = 0:5000:200000,
)
Plots.histogram(
    [MPC_result["power_utilization"][p] for p in DC.toppower_IDs],
    bins = 800000:1000:900000,
)

MPC_result["failpower_utilization"]

# Plots.histogram([
#     MPC_result["failpower_utilization"][(1, 1, p)]
#     for p in intersect(
#         setdiff(DC.room_power_map[1], DC.power_descendants_map[1]),
#         DC.toppower_IDs,
#     )
#     ],
#     bins = 20,
# )


all_results = Dict{String, Any}()
all_results["SSOA_objective"] = Float64[]
# all_results["SSOA_objective_precedence"] = Float64[]
all_results["MPC_objective"] = Float64[]
# all_results["MPC_objective_precedence"] = Float64[]

push!(all_results["MPC_objective"], MPC_result["objective"])
# push!(all_results["MPC_objective_precedence"], MPC_result["objective_precedence"])
for seed in 1:5
# for seed in [1]
# for seed in [2]
# for seed in 3:5
    SSOA_result = run_experiment(
        distr_dir,
        demand_fp,
        joinpath(result_dir, "SSOA_$seed"),
        ;
        write = true,
        strategy = "SSOA",
        discount_factor = discount_factor,
        with_precedence = with_precedence,
        obj_minimize_rooms = obj_minimize_rooms,
        obj_minimize_rows = obj_minimize_rows,
        obj_minimize_tilegroups = obj_minimize_tilegroups,
        obj_minimize_power_surplus = obj_minimize_power_surplus,
        obj_minimize_power_balance = obj_minimize_power_balance,
        interpolate_power = interpolate_power,
        interpolate_cooling = interpolate_cooling,
        placement_reward = placement_reward,
        placement_var_reward = placement_var_reward,
        room_mult = room_mult,
        row_mult = row_mult,
        tilegroup_penalty = tilegroup_penalty,
        power_surplus_penalty = power_surplus_penalty,
        power_balance_penalty = power_balance_penalty,
        S = 1,
        seed = seed,
        MIPGap = MIPGap,
        time_limit_sec_per_iteration = time_limit_sec_per_iteration,
        test_run = test_run,
        verbose = verbose,
        toppower_capacity = 9e5,
        midpower_capacity = 9e5 / 5,
        lowpower_capacity = 9e5 / (5 * 2),
        cooling_capacity = 30000.0,
    ) 
    push!(all_results["SSOA_objective"], SSOA_result["objective"])
    # push!(all_results["SSOA_objective_precedence"], SSOA_result["objective_precedence"])
end

all_results["SSOA_objective"]
all_results["MPC_objective"]
all_results["SSOA_objective_precedence"]
all_results["MPC_objective_precedence"]

MPC_result



x_MPCj = JSON.parsefile("$(@__DIR__)/results/MPC/x.json")
x_SSOA1j = JSON.parsefile("$(@__DIR__)/results/SSOA_1/x.json")
y_MPCj = JSON.parsefile("$(@__DIR__)/results/MPC/y.json")
y_SSOA1j = JSON.parsefile("$(@__DIR__)/results/SSOA_1/y.json")
T = 15
m = 10
d = length(DC.tilegroup_IDs)
R = length(DC.row_IDs)



x_MPC = zeros(Int, (T, m, d))
x_SSOA1 = zeros(Int, (T, m, d))
for k in keys(x_MPCj)
    mt = match(r"\((\d+), (\d+), (\d+)\)", k)
    x_MPC[parse(Int, mt.captures[1]), parse(Int, mt.captures[2]), parse(Int, mt.captures[3])] = x_MPCj[k]
end
for k in keys(x_SSOA1j)
    mt = match(r"\((\d+), (\d+), (\d+)\)", k)
    x_SSOA1[parse(Int, mt.captures[1]), parse(Int, mt.captures[2]), parse(Int, mt.captures[3])] = x_SSOA1j[k]
end

y_MPC = zeros(Int, (T, m, R))
y_SSOA1 = zeros(Int, (T, m, R))
for k in keys(y_MPCj)
    mt = match(r"\((\d+), (\d+), (\d+)\)", k)
    y_MPC[parse(Int, mt.captures[1]), parse(Int, mt.captures[2]), parse(Int, mt.captures[3])] = y_MPCj[k]
end
for k in keys(y_SSOA1j)
    mt = match(r"\((\d+), (\d+), (\d+)\)", k)
    y_SSOA1[parse(Int, mt.captures[1]), parse(Int, mt.captures[2]), parse(Int, mt.captures[3])] = y_SSOA1j[k]
end


sum(x_MPC[1, :, :], dims=1)
sum(x_SSOA1[1, :, :], dims=1)

sum(y_MPC[1, :, :], dims=1)
sum(y_SSOA1[1, :, :], dims=1)





all_results_noprecedence = Dict{String, Any}()
all_results_noprecedence["SSOA_objective"] = Float64[]
all_results_noprecedence["MPC_objective"] = Float64[]


MPC_result = run_experiment(
    distr_dir,
    demand_fp,
    result_dir,
    ;
    write = true,
    strategy = "MPC",
    discount_factor = discount_factor,
    with_precedence = with_precedence,
    obj_minimize_rooms = obj_minimize_rooms,
    obj_minimize_rows = obj_minimize_rows,
    obj_minimize_tilegroups = obj_minimize_tilegroups,
    obj_minimize_power_surplus = obj_minimize_power_surplus,
    obj_minimize_power_balance = obj_minimize_power_balance,
    placement_reward = placement_reward,
    placement_var_reward = placement_var_reward,
    room_mult = room_mult,
    row_mult = row_mult,
    tilegroup_penalty = tilegroup_penalty,
    power_surplus_penalty = power_surplus_penalty,
    power_balance_penalty = power_balance_penalty,
    S = 1,
    MIPGap = MIPGap,
    time_limit_sec_per_iteration = time_limit_sec_per_iteration,
    test_run = test_run,
    verbose = verbose,
)
push!(all_results_noprecedence["MPC_objective"], MPC_result["objective"])
for seed in 1:5
    SSOA_result = run_experiment(
        distr_dir,
        demand_fp,
        result_dir,
        ;
        write = true,
        strategy = "SSOA",
        discount_factor = discount_factor,
        with_precedence = with_precedence,
        obj_minimize_rooms = obj_minimize_rooms,
        obj_minimize_rows = obj_minimize_rows,
        obj_minimize_tilegroups = obj_minimize_tilegroups,
        obj_minimize_power_surplus = obj_minimize_power_surplus,
        obj_minimize_power_balance = obj_minimize_power_balance,
        placement_reward = placement_reward,
        placement_var_reward = placement_var_reward,
        room_mult = room_mult,
        row_mult = row_mult,
        tilegroup_penalty = tilegroup_penalty,
        power_surplus_penalty = power_surplus_penalty,
        power_balance_penalty = power_balance_penalty,
        S = 1,
        seed = seed,
        MIPGap = MIPGap,
        time_limit_sec_per_iteration = time_limit_sec_per_iteration,
        test_run = test_run,
        verbose = verbose,
    ) 
    push!(all_results_noprecedence["SSOA_objective"], SSOA_result["objective"])
end

all_results_noprecedence["SSOA_objective"]
all_results_noprecedence["MPC_objective"]






