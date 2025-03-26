
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
include("$(@__DIR__)/build_datacenter.jl")
include("$(@__DIR__)/simulate_batch.jl")
include("$(@__DIR__)/model.jl")
include("$(@__DIR__)/read_demand.jl")


### Filepaths ### 
fps = JSON.parsefile("$(@__DIR__)/../filepaths.json")
datacenter_dir = fps["datacenter_dir"]
distr_dir = fps["distr_dir"]
demand_fps = fps["demand_fps"]
run_ind = 5
demand_fp = demand_fps[run_ind]
### 

RCoeffs = RackPlacementCoefficients(
    # placement_reward = 0.0,
    # placement_var_reward = 20.0,
    discount_factor = 1.0,
)

DC = build_datacenter(datacenter_dir)
Sim = HistoricalDemandSimulator(distr_dir)
batches, batch_sizes = read_demand(demand_fp, RCoeffs.placement_reward, RCoeffs.placement_var_reward)

oracle_result = rack_placement_oracle(
    batches, batch_sizes, DC, 
    time_limit_sec = 300,
)
oracle_result["objective"]
oracle_precedence_result = rack_placement_oracle(
    batches, batch_sizes, DC, 
    time_limit_sec = 30,
    with_precedence = true,
)
oracle_precedence_result["objective"]
oracle_precedence_result["u"]

SSOA_precedence_result = rack_placement(
    DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SSOA", S = 1, seed = 0,
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

SSOA_precedence_result["all_results"][1]





SSOA_result = rack_placement(
    DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SSOA", S = 1, seed = 0,
    time_limit_sec_per_iteration = 100,
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
SSOA_r = postprocess_results(
    SSOA_result["all_results"], batch_sizes, DC, "SSOA"; 
    obj_minimize_rooms = false,
    obj_minimize_rows = true,
    obj_minimize_tilegroups = false,
    obj_minimize_power_surplus = true,
    obj_minimize_power_balance = true,
)
SSOA_result["objective"]

SAA_result = rack_placement(DC, Sim, RCoeffs, batches, batch_sizes, strategy = "SAA", S = 5, seed = 0)
MPC_result = rack_placement(DC, Sim, RCoeffs, batches, batch_sizes, strategy = "MPC")

SSOA_result["objective"]
SAA_result["objective"]
MPC_result["objective"]




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