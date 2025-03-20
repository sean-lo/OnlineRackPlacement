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
        "optimality_gap" => JuMP.relative_gap(model),
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
    RCoeffsD::RackPlacementCoefficientsDynamic,
    ;
    sim_batches::Union{Dict, Nothing} = nothing,
    sim_batch_sizes::Union{Dict, Nothing} = nothing,
    S::Int = 1,
    env::Union{Gurobi.Env, Nothing} = nothing,
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
    MIPGap::Float64 = 1e-4,
    time_limit_sec = 300,
)
    if isnothing(env)
        env = Gurobi.Env()
    end

    start_time = time()

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
    if obj_minimize_power_surplus
        @variable(model, Φ_now ≥ 0)
    end
    if obj_minimize_power_balance
        @variable(model, Ψ_now ≥ 0)
        @variable(model, Ω_now ≥ 0)
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

    # Space
    @expression(
        model, 
        space_now[j in DC.tilegroup_IDs],
        sum(
            x_fixed[(τ,i,j)]
            for τ in 1:t-1, i in 1:batch_sizes[τ]
                if (τ,i,j) in keys(x_fixed)
        )
        + sum(
            x_now[i,j]
            for i in 1:batch_sizes[t]
        )
    )
    if strategy == "SAA"
        @expression(
            model, 
            space_next[j in DC.tilegroup_IDs, s in 1:S],
            sum(
                x_next[(τ,s,i,j)]
                for τ in t+1:T, s in 1:S, i in 1:sim_batch_sizes[(τ,s)]
            )
        )
        @constraint(
            model, 
            [j in DC.tilegroup_IDs, s in 1:S],
            space_now[j] + space_next[j,s] ≤ DC.tilegroup_space_capacity[j]
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model, 
            space_next[j in DC.tilegroup_IDs],
            sum(
                x_next[(τ,i,j)]
                for τ in t+1:T, i in 1:sim_batch_sizes[τ]
            )
        )
        @constraint(
            model, 
            [j in DC.tilegroup_IDs],
            space_now[j] + space_next[j] ≤ DC.tilegroup_space_capacity[j]
        )
    else
        @constraint(
            model, 
            [j in DC.tilegroup_IDs],
            space_now[j] ≤ DC.tilegroup_space_capacity[j]
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
        @constraint(
            model, 
            [i in 1:batch_sizes[t]],
            sum(w_now[i,m] for m in DC.room_IDs) ≤ 1
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
        @constraint(
            model, 
            [i in 1:batch_sizes[t]],
            sum(v_now[i,j] for j in DC.tilegroup_IDs) ≤ 1
        )
    end

    if obj_minimize_power_surplus || obj_minimize_power_balance
        @expression(
            model, 
            toppower_pair_utilization[
                m in DC.room_IDs,
                (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
            ],
            sum(
                batches[τ]["power"][i] * x_fixed[(τ,i,j)] 
                for τ in 1:t-1, 
                    i in 1:batch_sizes[τ], 
                    j in intersect(
                        DC.power_tilegroups_map[p1],
                        DC.power_tilegroups_map[p2]
                    )
                    if (τ,i,j) in keys(x_fixed)
            )
            + sum(
                batches[t]["power"][i] * x_now[i,j]
                for i in 1:batch_sizes[t], j in intersect(
                    DC.power_tilegroups_map[p1],
                    DC.power_tilegroups_map[p2]
                )
            )
        )
    end

    if obj_minimize_power_surplus
        @constraint(
            model, 
            [
                m in DC.room_IDs,
                (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
            ],
            Φ_now ≥ toppower_pair_utilization[m, (p1, p2)] - DC.power_balanced_capacity[m]
        )
    end

    if obj_minimize_power_balance
        @constraint(
            model, 
            [
                m in DC.room_IDs,
                (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
            ],
            Ψ_now ≥ toppower_pair_utilization[m, (p1, p2)]
        )
        @constraint(
            model, 
            [
                m in DC.room_IDs,
                (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
            ],
            Ω_now ≤ toppower_pair_utilization[m, (p1, p2)]
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
    @expression(
        model, 
        current_reward, 
        sum(
            batches[t]["reward"][i] * y_now[i,r] 
            for i in 1:batch_sizes[t], r in DC.row_IDs
        )
    )
    if obj_minimize_rooms
        @expression(
            model, 
            room_penalty,
            - sum(
                RCoeffsD.room_penalties[m] * w_now[i,m] 
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
                RCoeffsD.row_penalties[r] * z_now[r]
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
                RCoeffsD.tilegroup_penalty * v_now[i,j]
                for i in 1:batch_sizes[t], j in DC.tilegroup_IDs
            )
        )
        add_to_expression!(current_reward, tilegroup_penalty)
    end
    if obj_minimize_power_surplus
        @expression(
            model, 
            power_surplus_penalty,
            - RCoeffsD.power_surplus_penalty * Φ_now
        )
        add_to_expression!(current_reward, power_surplus_penalty)
    end
    if obj_minimize_power_balance
        @expression(
            model, 
            power_balance_penalty,
            - RCoeffsD.power_balance_penalty * (Ψ_now - Ω_now)
        )
        add_to_expression!(current_reward, power_balance_penalty)
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
        @objective(model, Max, current_reward + RCoeffsD.discount_factor * future_assignment)
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
        "w" => JuMP.value.(w_now),
        "z" => JuMP.value.(z_now),
        "v" => JuMP.value.(v_now),
        "time_taken" => time() - start_time,
        "objective" => JuMP.objective_value(model),
        "optimality_gap" => JuMP.relative_gap(model),
        "current_reward" => JuMP.value(current_reward),
        "current_assignment" => JuMP.value(current_assignment),
        "space" => Dict(
            j => JuMP.value(space_now[j])
            for j in DC.tilegroup_IDs
        ),
        "cooling" => Dict(
            c => JuMP.value(cooling_now[c]) 
            for c in DC.cooling_IDs
        ),
        "power" => Dict(
            p => JuMP.value(power_now[p]) 
            for p in DC.power_IDs
        ),
        "failpower" => Dict(
            (p_, p) => JuMP.value(failpower_now[p_,p]) 
            for p_ in DC.toppower_IDs
                for p in setdiff(DC.power_IDs, DC.power_descendants_map[p_])
        ),
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
    if obj_minimize_power_surplus || obj_minimize_power_balance
        results["toppower_pair_utilization"] = Dict(
            (m, p1, p2) => JuMP.value(toppower_pair_utilization[m, (p1, p2)])
            for m in DC.room_IDs
                for (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
        )
    end
    if obj_minimize_power_surplus
        results["power_surplus_penalty"] = JuMP.value(power_surplus_penalty)
    end
    if obj_minimize_power_balance
        results["power_balance_penalty"] = JuMP.value(power_balance_penalty)
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
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
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

    RCoeffsD = RackPlacementCoefficientsDynamic(RCoeffs)
    RCoeffsD.row_penalties = Dict(r => RCoeffs.row_penalty for r in DC.row_IDs)
    RCoeffsD.room_penalties = Dict(m => RCoeffs.room_penalty for m in DC.room_IDs)
    RCoeffsD.room_penalties[first(DC.room_IDs)] = 0.0
    T = length(batches)

    x_fixed = Dict{Tuple{Int, Int, Int}, Int}()
    y_fixed = Dict{Tuple{Int, Int, Int}, Int}()
    all_results = Dict{String, Any}[]

    for t in 1:T

        # Simulate
        if strategy in ["SSOA", "SAA", "MPC"]
            sim_batches, sim_batch_sizes = simulate_batches(
                strategy, Sim, RCoeffsD,
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
            RCoeffsD,
            sim_batches = sim_batches,
            sim_batch_sizes = sim_batch_sizes,
            S = S,
            env = env,
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
        merge!(x_fixed, results["x"])
        merge!(y_fixed, results["y"])
        if verbose
            println("--------------------------------")
            println("Iteration $t of $T completed in $(results["time_taken"]) s.")
            println("Placed $(length(results["x"])) new demands ($(length(x_fixed)) total).")
            println("Current assignment:  $(results["current_assignment"])")
            if strategy in ["SSOA", "SAA", "MPC"]
                println("Future assignment:   $(results["future_assignment"])")
            end
            if obj_minimize_rooms
                @printf("Room penalty:        %.2f\n", results["room_penalty"])
            end
            if obj_minimize_rows
                @printf("Row penalty:         %.2f\n", results["row_penalty"])
            end
            if obj_minimize_tilegroups
                @printf("Tilegroup penalty:   %.2f\n", results["tilegroup_penalty"])
            end
            println("--------------------------------")
        end
        
        # Compute metrics for current iteration
        update_metrics!(DC, results)
        push!(all_results, deepcopy(results))

        # Update dynamic parameters
        update_dynamic_parameters!(RCoeffsD, DC, results)
    end

    return Dict(
        "x" => x_fixed,
        "y" => y_fixed,
        "time_taken" => time() - start_time,
        "objective" => sum(
            batches[t]["reward"][i] * y_fixed[(t,i,r)]
            for (t, i, r) in keys(y_fixed)
        ),
        "all_results" => all_results,
    )
end

function update_dynamic_parameters!(
    RCoeffsD::RackPlacementCoefficientsDynamic,
    DC::DataCenter,
    results::Dict{String, Any},
)
    
    RCoeffsD.row_penalties = Dict(
        r => (
            results["row_space_utilizations"][r] <= 0 ? 2.0 : (
                results["row_space_utilizations"][r] <= 10 ? 1.0 : 0.0
            )
        )
        for r in DC.row_IDs
    )
    RCoeffsD.room_penalties = Dict(
        m => (
            results["room_space_utilizations"][m] <= 0 ? 40.0 : (
                results["room_space_utilizations"][m] <= 0.3 * (
                    length(DC.room_rows_map[m]) * 20 # Number of tiles in room m
                ) ? 3.0 : 0.0
            )
        )
        for m in DC.room_IDs
    )
    RCoeffsD.room_penalties[first(DC.room_IDs)] = 0.0
    return 
end

function update_metrics!(
    DC::DataCenter,
    results::Dict{String, Any},
)
    results["row_space_utilizations"] = Dict{Int, Float64}(
        r => sum(
            results["space"][j] 
            for j in DC.row_tilegroups_map[r]
        )
        for r in DC.row_IDs
    )
    results["room_space_utilizations"] = Dict{Int, Float64}(
        m => sum(
            results["row_space_utilizations"][r]
            for r in DC.room_rows_map[m]
        )
        for m in DC.room_IDs
    )
    results["toppower_utilizations"] = Dict{Int, Float64}(
        p => results["power"][p]
        for p in DC.toppower_IDs
    )
    return 

end

function postprocess_results(
    all_results::Vector{Dict{String, Any}},
    DC::DataCenter,
    strategy::String,
    ;
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
)
    keys_totake = [
        "time_taken", "optimality_gap",
        "current_reward", 
        "current_assignment",
    ]
    if strategy in ["SSOA", "SAA", "MPC"]
        push!(keys_totake, "future_assignment")
    end
    if obj_minimize_rooms
        push!(keys_totake, "room_penalty")
    end
    if obj_minimize_rows
        push!(keys_totake, "row_penalty")
    end
    if obj_minimize_tilegroups
        push!(keys_totake, "tilegroup_penalty")
    end
    if obj_minimize_power_surplus
        push!(keys_totake, "power_surplus_penalty")
    end
    if obj_minimize_power_balance
        push!(keys_totake, "power_balance_penalty")
    end

    iteration_data = DataFrame(
        Dict(k => [all_results[t][k] for t in 1:length(all_results)]
        for k in keys_totake)
    )
    room_space_utilization_data = DataFrame(Dict(
        "$m" => [
            all_results[t]["room_space_utilizations"][m]
            for t in 1:length(all_results)
        ]
        for m in DC.room_IDs
    ))
    toppower_utilization_data = DataFrame(Dict(
        "$p" => [
            all_results[t]["toppower_utilizations"][p]
            for t in 1:length(all_results)
        ]
        for p in DC.toppower_IDs
    ))
    toppower_pair_utilization_data = DataFrame(Dict(
        "$(p1), $(p2)" => [
            all_results[t]["toppower_pair_utilization"][(m, p1, p2)]
            for t in 1:length(all_results)
        ]
        for m in DC.room_IDs
            for (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
    ))
    return Dict(
        "iteration_data" => iteration_data,
        "room_space_utilization_data" => room_space_utilization_data,
        "toppower_utilization_data" => toppower_utilization_data,
        "toppower_pair_utilization_data" => toppower_pair_utilization_data,
    )
end