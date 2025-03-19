function rack_placement_oracle(
    batches::Dict{Int, Dict{String, Any}},
    batch_sizes::Dict{Int, Int},
    DC::DataCenter,
    env::Union{Gurobi.Env, Nothing} = nothing,
    time_limit_sec = 300,
)
    if isnothing(env)
        env = Gurobi.Env()
    end

    start_time = time()
    T = length(batches)

    model = Model(() -> Gurobi.Optimizer(env))
    set_optimizer_attribute(model, "MIPGap", 1e-4)
    set_time_limit_sec(model, time_limit_sec)

    @variable(model, x[t in 1:T, i in 1:batch_sizes[t], j in DC.tilegroup_IDs] ≥ 0, Int)
    @variable(model, y[t in 1:T, i in 1:batch_sizes[t], r in DC.row_IDs] ≥ 0, Bin)

    # Assignment
    @constraint(
        model, 
        [t in 1:T, i in 1:batch_sizes[t]], 
        sum(y[t,i,:]) ≤ 1,
    )
    # Space
    @constraint(
        model, 
        [t in 1:T, i in 1:batch_sizes[t], r in DC.row_IDs], 
        sum(x[t,i,j] for j in DC.row_tilegroups_map[r]) 
        == y[t,i,r] * batches[t]["size"][i]
    )
    # Cooling
    @constraint(
        model, 
        [c in DC.cooling_IDs],
        sum(
            batches[t]["cooling"][i] * x[t,i,j]
            for t in 1:T, i in 1:batch_sizes[t], j in DC.cooling_tilegroups_map[c]
        )
        ≤ DC.cooling_capacity[c]
    )
    # Power
    @constraint(
        model, 
        [p in DC.power_IDs],
        sum(
            (batches[t]["power"][i] / 2) * x[t,i,j]
            for t in 1:T, i in 1:batch_sizes[t], j in DC.power_tilegroups_map[p]
        )
        ≤ DC.power_capacity[p]
    )
    # Failover power
    @constraint(
        model, 
        [p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])],
        sum(
            (batches[t]["power"][i] / 2) * (
                sum(
                    x[t,i,j]
                    for j in DC.power_tilegroups_map[p]
                )
                + sum(
                    x[t,i,j]
                    for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                )
            )
            for t in 1:T, i in 1:batch_sizes[t]
        )
        ≤ DC.failpower_capacity[p]
    )
    @objective(
        model,
        Max,
        sum(
            batches[t]["reward"][i] * y[t,i,r]
            for t in 1:T, i in 1:batch_sizes[t], r in DC.row_IDs
        )
    )
    optimize!(model)
    
    x_result = Dict(
        (t, i, j) => round(JuMP.value(x[t,i,j]))
        for (t, i, j) in keys(x.data)
            if round(JuMP.value(x[t,i,j])) > 0
    )
    y_result = Dict(
        (t, i, r) => round(JuMP.value(y[t,i,r]))
        for (t, i, r) in keys(y.data)
            if round(JuMP.value(y[t,i,r])) > 0
    )
    return Dict(
        "x" => x_result,
        "y" => y_result,
        "objective" => JuMP.objective_value(model),
        "time_taken" => time() - start_time,
    )
end


function build_solve_incremental_model(
    x_fixed::Dict{Tuple{Int, Int, Int}, Int},
    y_fixed::Dict{Tuple{Int, Int, Int}, Int},
    DC::DataCenter,
    t::Int,
    T::Int,
    batches::Dict{Int, Dict{String, Any}},
    batch_sizes::Dict{Int, Int},
    strategy::String,
    RCoeffs::RackPlacementCoefficients,
    ;
    sim_batches::Union{Dict, Nothing} = nothing,
    sim_batch_sizes::Union{Dict, Nothing} = nothing,
    S::Int = 1,
    env::Union{Gurobi.Env, Nothing} = nothing,
    obj_minimize_rooms::Bool = false,
    obj_minimize_rows::Bool = false,
    obj_minimize_tilegroups::Bool = false,
    MIPGap::Float64 = 1e-4,
    time_limit_sec = 300,
)
    if isnothing(env)
        env = Gurobi.Env()
    end

    model = Model(() -> Gurobi.Optimizer(env))
    set_optimizer_attribute(model, "MIPGap", MIPGap)
    set_time_limit_sec(model, time_limit_sec)

    @variable(model, x_now[i in 1:batch_sizes[t], j in DC.tilegroup_IDs] ≥ 0, Int)
    @variable(model, y_now[i in 1:batch_sizes[t], r in DC.row_IDs], Bin)
    if obj_minimize_rooms
        @variable(model, w_now[i in 1:batch_sizes[t], m in DC.room_IDs], Bin)
    end
    if obj_minimize_rows
        @variable(model, z_now[r in DC.row_IDs], Bin)
    end
    if obj_minimize_tilegroups
        @variable(model, v_now[i in 1:batch_sizes[t], j in DC.tilegroup_IDs], Bin)
    end
    if strategy == "SAA"
        @variable(model, x_next[τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)], j in DC.tilegroup_IDs] ≥ 0, Int)
        @variable(model, y_next[τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)], r in DC.row_IDs], Bin)
    elseif strategy in ["SSOA", "MPC"]
        @variable(model, x_next[τ in t+1:T, i in 1:sim_batch_sizes[τ], j in DC.tilegroup_IDs] ≥ 0, Int)
        @variable(model, y_next[τ in t+1:T, i in 1:sim_batch_sizes[τ], r in DC.row_IDs], Bin)
    end

    # Assignment
    @constraint(
        model, 
        [i in 1:batch_sizes[t]], 
        sum(y_now[i,:]) ≤ 1,
    )
    @constraint(
        model, 
        [i in 1:batch_sizes[t], r in DC.row_IDs], 
        sum(x_now[i,j] for j in DC.row_tilegroups_map[r]) 
        == y_now[i,r] * batches[t]["size"][i]
    )
    if strategy == "SAA"
        @constraint(
            model, 
            [τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)]], 
            sum(y_next[τ,s,i,:]) ≤ 1,
        )
        @constraint(
            model, 
            [τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)], r in DC.row_IDs], 
            sum(x_next[τ,s,i,j] for j in DC.row_tilegroups_map[r]) 
            == y_next[τ,s,i,r] * batches[t]["size"][i]
        )
    elseif strategy in ["SSOA", "MPC"]
        @constraint(
            model, 
            [τ in t+1:T, i in 1:sim_batch_sizes[τ]], 
            sum(y_next[τ,i,:]) ≤ 1,
        )
        @constraint(
            model, 
            [τ in t+1:T, i in 1:sim_batch_sizes[τ], r in DC.row_IDs], 
            sum(x_next[τ,i,j] for j in DC.row_tilegroups_map[r]) 
            == y_next[τ,i,r] * batches[t]["size"][i]
        )
    end

    # Cooling
    @expression(
        model,
        cooling_now[c in DC.cooling_IDs],
        sum(
            batches[τ]["cooling"][i] * x_fixed[(τ,i,j)]
            for τ in 1:t-1, i in 1:batch_sizes[τ], j in DC.cooling_tilegroups_map[c]
                if (τ,i,j) in keys(x_fixed)
        )
        + sum(
            batches[t]["cooling"][i] * x_now[i,j]
            for i in 1:batch_sizes[t], j in DC.cooling_tilegroups_map[c]
        )
    )
    if strategy == "SAA"
        @expression(
            model,
            cooling_next[c in DC.cooling_IDs, s in 1:S],
            sum(
                sim_batches[(τ,s)]["cooling"][i] * x_next[τ,s,i,j]
                for τ in t+1:T, i in 1:sim_batch_sizes[(τ,s)], j in DC.cooling_tilegroups_map[c]
            )
        )
        @constraint(
            model, 
            [c in DC.cooling_IDs, s in 1:S],
            cooling_now[c] + cooling_next[c,s] ≤ DC.cooling_capacity[c]
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model,
            cooling_next[c in DC.cooling_IDs],
            sum(
                sim_batches[τ]["cooling"][i] * x_next[τ,i,j]
                for τ in t+1:T, i in 1:sim_batch_sizes[τ], j in DC.cooling_tilegroups_map[c]
            )
        )
        @constraint(
            model, 
            [c in DC.cooling_IDs],
            cooling_now[c] + cooling_next[c] ≤ DC.cooling_capacity[c]
        )
    else
        @constraint(
            model, 
            [c in DC.cooling_IDs],
            cooling_now[c] ≤ DC.cooling_capacity[c]
        )
    end

    # Power 
    @expression(
        model,
        power_now[p in DC.power_IDs],
        sum(
            (batches[τ]["power"][i] / 2) * x_fixed[(τ,i,j)]
            for τ in 1:t-1, i in 1:batch_sizes[τ], j in DC.power_tilegroups_map[p]
                if (τ,i,j) in keys(x_fixed)
        )
        + sum(
            (batches[t]["power"][i] / 2) * x_now[i,j]
            for i in 1:batch_sizes[t], j in DC.power_tilegroups_map[p]
        )
    )
    if strategy == "SAA"
        @expression(
            model,
            power_next[p in DC.power_IDs, s in 1:S],
            sum(
                (sim_batches[(τ,s)]["power"][i] / 2) * x_next[τ,s,i,j]
                for τ in t+1:T, i in 1:sim_batch_sizes[(τ,s)], j in DC.power_tilegroups_map[p]
            )
        )
        @constraint(
            model, 
            [p in DC.power_IDs, s in 1:S],
            power_now[p] + power_next[p,s] ≤ DC.power_capacity[p]
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model,
            power_next[p in DC.power_IDs],
            sum(
                (sim_batches[τ]["power"][i] / 2) * x_next[τ,i,j]
                for τ in t+1:T, i in 1:sim_batch_sizes[τ], j in DC.power_tilegroups_map[p]
            )
        )
        @constraint(
            model, 
            [p in DC.power_IDs],    
            power_now[p] + power_next[p] ≤ DC.power_capacity[p] 
        )
    else
        @constraint(
            model, 
            [p in DC.power_IDs],
            power_now[p] ≤ DC.power_capacity[p]
        )
    end

    # Failover power
    @expression(
        model,
        failpower_now[p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])],
        sum(
            (batches[τ]["power"][i] / 2) * (
                sum(
                    x_fixed[(τ,i,j)]
                    for j in DC.power_tilegroups_map[p]
                        if (τ,i,j) in keys(x_fixed)
                )
                + sum(
                    x_fixed[(τ,i,j)]
                    for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                        if (τ,i,j) in keys(x_fixed)
                )
            )
            for τ in 1:t-1, i in 1:batch_sizes[τ]
        )
        + sum(
            (batches[t]["power"][i] / 2) * (
                sum(
                    x_now[i,j]
                    for j in DC.power_tilegroups_map[p]
                )
                + sum(
                    x_now[i,j]
                    for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                )
            )
            for i in 1:batch_sizes[t]
        )
    )
    if strategy == "SAA"
        @expression(
            model,
            failpower_next[p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_]), s in 1:S],
            sum(
                (sim_batches[(τ,s)]["power"][i] / 2) * (
                    sum(
                        x_next[τ,s,i,j]
                        for j in DC.power_tilegroups_map[p]
                    )
                    + sum(
                        x_next[τ,s,i,j]
                        for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                    )
                )
                for τ in t+1:T, i in 1:sim_batch_sizes[(τ,s)]
            )
        )
        @constraint(
            model,
            [p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_]), s in 1:S],
            failpower_now[p_,p] + failpower_next[p_,p,s] ≤ DC.failpower_capacity[p]
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model,
            failpower_next[p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])],
            sum(
                (sim_batches[τ]["power"][i] / 2) * (
                    sum(
                        x_next[τ,i,j]
                        for j in DC.power_tilegroups_map[p]
                    )
                    + sum(
                        x_next[τ,i,j]
                        for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                    )
                )
                for τ in t+1:T, i in 1:sim_batch_sizes[τ]
            )
        )
        @constraint(
            model,
            [p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])],
            failpower_now[p_,p] + failpower_next[p_,p] ≤ DC.failpower_capacity[p]
        )
    else
        @constraint(
            model,
            [p_ in DC.toppower_IDs, p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])],
            failpower_now[p_,p] ≤ DC.failpower_capacity[p]
        )
    end

    if obj_minimize_rooms
        @constraint(
            model, 
            [i in 1:batch_sizes[t], r in DC.row_IDs],
            y_now[i,r] ≤ w_now[i, DC.row_room_map[r]]
        )
    end

    if obj_minimize_rows
        @constraint(
            model, 
            [i in 1:batch_sizes[t], r in DC.row_IDs],
            y_now[i,r] ≤ z_now[r]
        )
    end

    if obj_minimize_tilegroups
        @constraint(
            model,
            [i in 1:batch_sizes[t], j in DC.tilegroup_IDs],
            x_now[i,j] ≤ batches[t]["size"][i] * v_now[i,j]
        )
    end

    @expression(
        model, 
        current_assignment, 
        sum(
            batches[t]["reward"][i] * y_now[i,r] 
            for i in 1:batch_sizes[t], r in DC.row_IDs
        )
    )
    current_reward = current_assignment
    if obj_minimize_rooms
        @expression(
            model, 
            room_penalty,
            - sum(
                RCoeffs.room_penalty * w_now[i,m] 
                for i in 1:batch_sizes[t], m in DC.room_IDs
            )
        )
        add_to_expression!(current_reward, room_penalty)
    end
    if obj_minimize_rows
        @expression(
            model,
            row_penalty,
            - sum(
                RCoeffs.row_penalty * z_now[r]
                for r in DC.row_IDs
            )
        )
        add_to_expression!(current_reward, row_penalty)
    end
    if obj_minimize_tilegroups
        @expression(
            model, 
            tilegroup_penalty, 
            - sum(
                RCoeffs.tilegroup_penalty * v_now[i,j]
                for i in 1:batch_sizes[t], j in DC.tilegroup_IDs
            )
        )
        add_to_expression!(current_reward, tilegroup_penalty)
    end

    if strategy == "SAA"
        @expression(
            model,
            future_assignment,
            sum(
                sim_batches[(τ,s)]["reward"][i] * y_next[τ,s,i,r]   
                for τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)], r in DC.row_IDs
            ) / S
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model,
            future_assignment,
            sum(
                sim_batches[τ]["reward"][i] * y_next[τ,i,r]   
                for τ in t+1:T, i in 1:sim_batch_sizes[τ], r in DC.row_IDs
            )
        )
    end

    if strategy == "myopic"
        @objective(model, Max, current_reward)
    elseif strategy in ["SSOA", "MPC", "SAA"]
        @objective(model, Max, current_reward + RCoeffs.discount_factor * future_assignment)
    end
    
    optimize!(model)
    
    x_fixed_new = Dict{Tuple{Int, Int, Int}, Int}()
    y_fixed_new = Dict{Tuple{Int, Int, Int}, Int}()
    for i in 1:batch_sizes[t], j in DC.tilegroup_IDs
        val = round(JuMP.value(x_now[i,j]))
        if val > 0
            x_fixed_new[(t,i,j)] = val
        end
    end
    for i in 1:batch_sizes[t], r in DC.row_IDs
        val = round(JuMP.value(y_now[i,r]))
        if val == 1
            y_fixed_new[(t,i,r)] = val
        end
    end

    results = Dict(
        "x" => x_fixed_new,
        "y" => y_fixed_new,
        "objective" => JuMP.objective_value(model),
        "current_reward" => JuMP.value(current_reward),
        "current_assignment" => JuMP.value(current_assignment),
    )
    if strategy in ["SSOA", "SAA", "MPC"]
        results["future_assignment"] = JuMP.value(future_assignment)
    end
    if obj_minimize_rooms
        results["room_penalty"] = JuMP.value(room_penalty)
    end
    if obj_minimize_rows
        results["row_penalty"] = JuMP.value(row_penalty)
    end
    if obj_minimize_tilegroups
        results["tilegroup_penalty"] = JuMP.value(tilegroup_penalty)
    end
    return results
end

function rack_placement(
    DC::DataCenter,
    Sim::HistoricalDemandSimulator,
    RCoeffs::RackPlacementCoefficients,
    batches::Dict{Int, Dict{String, Any}},
    batch_sizes::Dict{Int, Int}, 
    ;
    env::Union{Gurobi.Env, Nothing} = nothing,
    strategy::String = "SSOA",
    S::Int = 1, # Number of sample paths
    seed::Union{Int, Nothing} = nothing,
    MIPGap::Float64 = 1e-4,
    time_limit_sec = 0,
    time_limit_sec_per_iteration = 60,
    verbose::Bool = true,
)
    if isnothing(env)
        env = Gurobi.Env()
    end
    if isnothing(seed)
        seed = abs(Random.rand(Int))
    end
    if !(strategy in ["myopic", "SSOA", "SAA", "MPC"])
        error("ArgumentError: strategy = $strategy not recognized.")
    end

    start_time = time()

    T = length(batches)

    x_fixed = Dict{Tuple{Int, Int, Int}, Int}()
    y_fixed = Dict{Tuple{Int, Int, Int}, Int}()
    time_taken = Float64[]

    for t in 1:T

        # Simulate
        if strategy in ["SSOA", "SAA", "MPC"]
            sim_batches, sim_batch_sizes = simulate_batches(
                strategy, Sim, RCoeffs,
                t, T,
                batch_sizes, S,
            )
        else
            sim_batches, sim_batch_sizes = nothing, nothing
        end

        # Optimize incremental model
        results = build_solve_incremental_model(
            x_fixed,
            y_fixed,
            DC,
            t,
            T,
            batches,
            batch_sizes,
            strategy,
            RCoeffs,
            sim_batches = sim_batches,
            sim_batch_sizes = sim_batch_sizes,
            S = S,
            env = env,
            MIPGap = MIPGap,
            time_limit_sec = max(
                time_limit_sec_per_iteration,
                time_limit_sec - (time() - start_time),
            )
        )
        merge!(x_fixed, results["x"])
        merge!(y_fixed, results["y"])
        push!(time_taken, time() - start_time)
        if verbose
            println("--------------------------------")
            println("Iteration $t of $T completed in $(time_taken[t]) s.")
            println("Placed $(length(x_fixed_new)) new demands ($(length(x_fixed)) total).")
            println("--------------------------------")
        end
    end

    return Dict(
        "x" => x_fixed,
        "y" => y_fixed,
        "time_taken" => time_taken,
        "objective" => sum(
            batches[t]["reward"][i] * y_fixed[(t,i,r)]
            for (t, i, r) in keys(y_fixed)
        ),
    )
end