
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
# datacenter_dir = fps["datacenter_dir"]
demand_fps = fps["demand_fps"]
run_ind = 1
distr_dir = fps["distr_dir"]
demand_fp = demand_fps[run_ind]
### 

include("$(@__DIR__)/experiment.jl")

# Parameters
use_batching = false
batch_size = 10
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
discount_factor = 0.1
with_precedence = false
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
time_limit_sec_per_iteration = 600
test_run = false
verbose = true
result_dir = "$(@__DIR__)/results/"


Sim = HistoricalDemandSimulator(
    distr_dir,
    interpolate_power = interpolate_power,
    interpolate_cooling = interpolate_cooling,
)

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
)
MPC_result["objective"]
MPC_result["power_utilization"]


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
all_results["SSOA_objective_precedence"] = Float64[]
all_results["MPC_objective"] = Float64[]
all_results["MPC_objective_precedence"] = Float64[]

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








RCoeffs = RackPlacementCoefficients(
    # placement_reward = 0.0,
    # placement_var_reward = 20.0,
    discount_factor = discount_factor,
)
# DC = read_datacenter(datacenter_dir)
DC = build_datacenter()
Sim = HistoricalDemandSimulator(distr_dir, interpolate_power = false, interpolate_cooling = false)
batches, batch_sizes = read_demand(
    demand_fp, RCoeffs.placement_reward, RCoeffs.placement_var_reward,
    false, 10,
)

oracle_result = rack_placement_oracle(
    batches, batch_sizes, DC, 
    MIPGap = 1e-3,
    time_limit_sec = 300,
)
oracle_result["objective"]
oracle_precedence_result = rack_placement_oracle(
    batches, batch_sizes, DC, 
    # time_limit_sec = 30,
    MIPGap = 1e-3,
    with_precedence = true,
)
oracle_precedence_result["objective"]

SSOA_precedence_result = rack_placement(
    DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SSOA", S = 1, seed = 2,
    MIPGap = 1e-3,
    time_limit_sec_per_iteration = 10,
    with_precedence = true,
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
SSOA_precedence_result["objective"]
SSOA_precedence_r = postprocess_results(
    SSOA_precedence_result["all_results"], batch_sizes, DC, "SSOA"; 
    with_precedence = true,
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
SSOA_precedence_r["objective_precedence"]


[SSOA_precedence_result["all_results"][end]["power"][i] for i in DC.toppower_IDs]
[SSOA_precedence_result["all_results"][end]["power"][i] for i in 29:100] |> mean

using Plots

Plots.histogram(
    [SSOA_precedence_result["all_results"][end]["cooling"][i] for i in DC.cooling_IDs],
    bins = 100,
)
Plots.histogram(
    [SSOA_precedence_result["all_results"][end]["space"][i] for i in DC.tilegroup_IDs],
)

Plots.histogram(
    [
        sum([SSOA_precedence_result["all_results"][end]["space"][j] for j in DC.row_tilegroups_map[r]])
        for r in DC.row_IDs
    ],
    bins=21,
)



MPC_precedence_result = rack_placement(
    DC, Sim, RCoeffs, batches, batch_sizes, strategy = "MPC", S = 1, seed = 0,
    MIPGap = 1e-3,
    time_limit_sec_per_iteration = 10,
    with_precedence = true,
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
MPC_precedence_result["objective"]

MPC_precedence_r = postprocess_results(
    MPC_precedence_result["all_results"], batch_sizes, DC, "MPC"; 
    with_precedence = true,
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
MPC_precedence_r["objective_precedence"]

simulate_batches("MPC", Sim, RCoeffs.placement_reward, RCoeffs.placement_var_reward, 1, length(batch_sizes), batch_sizes, S = 1, seed = 0)
mean_demand(Sim, RCoeffs.placement_reward, RCoeffs.placement_var_reward, 1)

distr_data = read_CSVs_from_dir(distr_dir)
sort!(distr_data["size"], [:size])
size_quantiles = cumsum(distr_data["size"][!, "frequency"])
size_quantiles = round.(size_quantiles, digits = 4)
size_quantiles[end] = 1.0
size_mean = sum(distr_data["size"][!, "size"] .* distr_data["size"][!, "frequency"])



Sim.size_mean

SSOA_result["time_taken"][end]
SAA_result["time_taken"][end]
MPC_result["time_taken"][end]



include("$(@__DIR__)/model.jl")

strategy = "SSOA"
S = 1
time_limit_sec_per_iteration = 20
time_limit_sec = 0
start_time = time()
env = Gurobi.Env()
MIPGap = 1e-4
with_precedence = true
obj_minimize_rooms = false
obj_minimize_rows = true
obj_minimize_tilegroups = false
obj_minimize_power_surplus = true
obj_minimize_power_balance = true


RCoeffsD = RackPlacementCoefficientsDynamic(RCoeffs)
RCoeffsD.row_penalties = Dict(r => RCoeffs.row_penalty for r in DC.row_IDs)
RCoeffsD.room_penalties = Dict(m => RCoeffs.room_penalty for m in DC.room_IDs)
RCoeffsD.room_penalties[first(DC.room_IDs)] = 0.0
T = length(batches)

all_results = Dict{String, Any}[]
x_fixed = Dict{Tuple{Int, Int, Int}, Int}()
y_fixed = Dict{Tuple{Int, Int, Int}, Int}()
u_fixed = Dict{Tuple{Int, Int}, Int}()

t = 1


sim_batches = simulate_batches(
    strategy, Sim, 
    RCoeffsD.placement_reward,
    RCoeffsD.placement_var_reward,
    t, T,
    batch_sizes,
    ;
    S = S,
    seed = 0,
)
results = build_solve_incremental_model(
    x_fixed,
    y_fixed,
    DC,
    t,
    T,
    batches,
    batch_sizes,
    strategy,
    RCoeffsD,
    sim_batches = sim_batches,
    S = S,
    env = env,
    with_precedence = with_precedence,
    u_fixed = u_fixed,
    obj_minimize_rooms = obj_minimize_rooms,
    obj_minimize_rows = obj_minimize_rows,
    obj_minimize_tilegroups = obj_minimize_tilegroups,
    obj_minimize_power_surplus = obj_minimize_power_surplus,
    obj_minimize_power_balance = obj_minimize_power_balance,
    MIPGap = MIPGap,
    time_limit_sec = max(
        time_limit_sec_per_iteration,
        time_limit_sec - (time() - start_time),
    )
)

results["u"]
results["current_assignment"]
results["u_next"]







model = Model(() -> Gurobi.Optimizer(env))
set_optimizer_attribute(model, "MIPGap", MIPGap)
set_time_limit_sec(model, time_limit_sec)

@variable(model, x_now[i in 1:batch_sizes[t], j in DC.tilegroup_IDs] ≥ 0, Int)
@variable(model, y_now[i in 1:batch_sizes[t], r in DC.row_IDs], Bin)
@variable(model, u_now[i in 1:batch_sizes[t]], Bin)
if obj_minimize_rooms
    @variable(model, w_now[i in 1:batch_sizes[t], m in DC.room_IDs], Bin)
end
if obj_minimize_rows
    @variable(model, z_now[r in DC.row_IDs], Bin)
end
if obj_minimize_tilegroups
    @variable(model, v_now[i in 1:batch_sizes[t], j in DC.tilegroup_IDs], Bin)
end
if obj_minimize_power_surplus
    @variable(model, Φ_now ≥ 0)
end
if obj_minimize_power_balance
    @variable(model, Ψ_now ≥ 0)
    @variable(model, Ω_now ≥ 0)
end
if strategy == "SAA"
    @variable(model, x_next[τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ], j in DC.tilegroup_IDs] ≥ 0, Int)
    @variable(model, y_next[τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ], r in DC.row_IDs], Bin)
    @variable(model, u_next[τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ]], Bin)
elseif strategy in ["SSOA", "MPC"]
    @variable(model, x_next[τ in t+1:T, i in 1:batch_sizes[τ], j in DC.tilegroup_IDs] ≥ 0, Int)
    @variable(model, y_next[τ in t+1:T, i in 1:batch_sizes[τ], r in DC.row_IDs], Bin)
    @variable(model, u_next[τ in t+1:T, i in 1:batch_sizes[τ]], Bin)
end

@constraint(
    model, 
    [i in 1:batch_sizes[t]], 
    sum(y_now[i,:]) ≤ 1,
)