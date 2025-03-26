function rack_placement_oracle(
    batches::Vector{<:Dict{String, <:Array}},
    batch_sizes::Vector{Int},
    DC::DataCenter,
    ;
    env::Union{Gurobi.Env, Nothing} = nothing,
    with_precedence::Bool = false,
    time_limit_sec = 300,
    MIPGap::Float64 = 1e-4,
)
    if isnothing(env)
        env = Gurobi.Env()
    end

    start_time = time()
    T = length(batches)

    model = Model(() -> Gurobi.Optimizer(env))
    set_optimizer_attribute(model, "MIPGap", MIPGap)
    set_time_limit_sec(model, time_limit_sec)

    @variable(model, x[t in 1:T, i in 1:batch_sizes[t], j in DC.tilegroup_IDs] ≥ 0, Int)
    @variable(model, y[t in 1:T, i in 1:batch_sizes[t], r in DC.row_IDs] ≥ 0, Bin)
    @variable(model, u[t in 1:T, i in 1:batch_sizes[t]] ≥ 0, Bin) # Helper

    # Assignment
    @constraint(
        model, 
        [t in 1:T, i in 1:batch_sizes[t]],
        sum(y[t,i,:]) == u[t,i]
    )
    @constraint(
        model, 
        [t in 1:T, i in 1:batch_sizes[t]], 
        u[t,i] ≤ 1,
    )

    # Linking
    @constraint(
        model, 
        [t in 1:T, i in 1:batch_sizes[t], r in DC.row_IDs], 
        sum(x[t,i,j] for j in DC.row_tilegroups_map[r]) 
        == y[t,i,r] * batches[t]["size"][i]
    )

    # Precedence
    if with_precedence
        @constraint(
            model,
            [t in 1:T-1, i in 1:batch_sizes[t], i_ in 1:batch_sizes[t+1]],
            u[t,i] ≥ u[t+1,i_]
        )
    end

    # Space
    @constraint(
        model,
        [j in DC.tilegroup_IDs],
        sum(x[t,i,j] for t in 1:T, i in 1:batch_sizes[t])
        ≤ DC.tilegroup_space_capacity[j]
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
    u_result = Dict(
        (t, i) => round(JuMP.value(u[t,i]))
        for (t, i) in keys(u.data)
            if round(JuMP.value(u[t,i])) > 0
    )
    return Dict(
        "x" => x_result,
        "y" => y_result,
        "u" => u_result,
        "objective" => JuMP.objective_value(model),
        "demands_placed" => sum(values(y_result)),
        "racks_placed" => sum(values(x_result)),
        "optimality_gap" => JuMP.relative_gap(model),
        "time_taken" => time() - start_time,
    )
end


function postprocess_results_oracle(
    DC::DataCenter,
    results::Dict{String, Any},
    batches::Vector{<:Dict{String, <:Array}},
    ;
)
    T = length(batches)
    room_space_utilization_data = DataFrame(Dict(
        "$m" => [
            sum(
                results["x"][(t,i,j)]
                for (t,i,j) in keys(results["x"])
                    if j in DC.room_tilegroups_map[m]
            )
        ]
        for m in DC.room_IDs
    ))
    toppower_utilization_data = DataFrame(Dict(
        "$p" => [
            sum(
                (batches[t]["power"][i] / 2) * results["x"][(t,i,j)]
                for (t,i,j) in keys(results["x"])
                    if j in DC.power_tilegroups_map[p]
            )
        ] 
        for p in DC.toppower_IDs
    ))
    return Dict(
        "room_space_utilization_data" => room_space_utilization_data,
        "toppower_utilization_data" => toppower_utilization_data,
        "toppower_utilization" => (
            toppower_utilization_data[end, :] |> sum
        ) / (
            sum(DC.power_capacity[p] for p in DC.toppower_IDs)
        ),
    )
end


function build_solve_incremental_model(
    x_fixed::Dict{Tuple{Int, Int, Int}, Int},
    y_fixed::Dict{Tuple{Int, Int, Int}, Int},
    DC::DataCenter,
    t::Int,
    T::Int,
    batches::Vector{<:Dict{String, <:Array}},
    batch_sizes::Vector{Int},
    strategy::String,
    RCoeffsD::RackPlacementCoefficientsDynamic,
    ;
    sim_batches::Union{Vector{<:Dict{String, <:Array}}, Nothing} = nothing,
    S::Int = 1,
    env::Union{Gurobi.Env, Nothing} = nothing,
    with_precedence::Bool = false,
    u_fixed::Union{Dict{Tuple{Int, Int}, Int}, Nothing} = nothing,
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
    MIPGap::Float64 = 1e-4,
    time_limit_sec = 300,
    verbose::Bool = false,
)
    if isnothing(env)
        env = Gurobi.Env()
    end


    verbose && println("Building model for iteration $t of $T:")
    start_time = time()

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

    # Assignment
    @constraint(
        model, 
        [i in 1:batch_sizes[t]],
        sum(y_now[i,:]) == u_now[i]
    )
    @constraint(
        model, 
        [i in 1:batch_sizes[t]], 
        u_now[i] ≤ 1,
    )
    if strategy == "SAA"
        @constraint(
            model, 
            [τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ]],
            sum(y_next[τ,s,i,:]) == u_next[τ,s,i],
        )
        @constraint(
            model,
            [τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ]],
            u_next[τ,s,i] ≤ 1,
        )
    elseif strategy in ["SSOA", "MPC"]
        @constraint(
            model, 
            [τ in t+1:T, i in 1:batch_sizes[τ]],
            sum(y_next[τ,i,:]) == u_next[τ,i],
        )
        @constraint(
            model, 
            [τ in t+1:T, i in 1:batch_sizes[τ]],
            u_next[τ,i] ≤ 1,
        )
    end

    # Linking
    @constraint(
        model, 
        [i in 1:batch_sizes[t], r in DC.row_IDs], 
        sum(x_now[i,j] for j in DC.row_tilegroups_map[r]) 
        == y_now[i,r] * batches[t]["size"][i]
    )
    if strategy == "SAA"
        @constraint(
            model, 
            [τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ], r in DC.row_IDs], 
            sum(x_next[τ,s,i,j] for j in DC.row_tilegroups_map[r]) 
            == y_next[τ,s,i,r] * batches[t]["size"][i]
        )
    elseif strategy in ["SSOA", "MPC"]
        @constraint(
            model, 
            [τ in t+1:T, i in 1:batch_sizes[τ], r in DC.row_IDs], 
            sum(x_next[τ,i,j] for j in DC.row_tilegroups_map[r]) 
            == y_next[τ,i,r] * batches[t]["size"][i]
        )
    end
    
    # Precedence
    if with_precedence
        # Comment: do not implement this constraint, 
        # since we still want to place the batches after the first drop
        # if t > 1
        #     @constraint(
        #         model,
        #         [i in 1:batch_sizes[t-1], i_ in 1:batch_sizes[t]],
        #         u_fixed[(t-1,i)] ≥ u_now[i_]
        #     )
        # end
        if strategy == "SAA" && t < T
            @constraint(
                model,
                precedence_now[s in 1:S, i in 1:batch_sizes[t], i_ in 1:batch_sizes[t+1]],
                u_now[i] ≥ u_next[t+1,s,i_]
            )
            @constraint(
                model,
                precedence_next[τ in t+1:T-1, s in 1:S, i in 1:batch_sizes[τ], i_ in 1:batch_sizes[τ+1]],
                u_next[τ,s,i] ≥ u_next[τ+1,s,i_]
            )
        elseif strategy in ["SSOA", "MPC"] && t < T
            @constraint(
                model,
                precedence_now[i in 1:batch_sizes[t], i_ in 1:batch_sizes[t+1]],
                u_now[i] ≥ u_next[t+1,i_]
            )
            @constraint(
                model,
                # Only implement until τ = T-1
                precedence_next[τ in t+1:T-1, i in 1:batch_sizes[τ], i_ in 1:batch_sizes[τ+1]],
                u_next[τ,i] ≥ u_next[τ+1,i_]
            )
        end
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
                for τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ]
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
                for τ in t+1:T, i in 1:batch_sizes[τ]
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
                sim_batches[τ-t]["cooling"][s,i] * x_next[τ,s,i,j]
                for τ in t+1:T, i in 1:batch_sizes[τ], j in DC.cooling_tilegroups_map[c]
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
                sim_batches[τ-t]["cooling"][i] * x_next[τ,i,j]
                for τ in t+1:T, i in 1:batch_sizes[τ], j in DC.cooling_tilegroups_map[c]
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
                (sim_batches[τ-t]["power"][s,i] / 2) * x_next[τ,s,i,j]
                for τ in t+1:T, i in 1:batch_sizes[τ], j in DC.power_tilegroups_map[p]
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
                (sim_batches[τ-t]["power"][i] / 2) * x_next[τ,i,j]
                for τ in t+1:T, i in 1:batch_sizes[τ], j in DC.power_tilegroups_map[p]
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
                (sim_batches[τ-t]["power"][s,i] / 2) * (
                    sum(
                        x_next[τ,s,i,j]
                        for j in DC.power_tilegroups_map[p]
                    )
                    + sum(
                        x_next[τ,s,i,j]
                        for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                    )
                )
                for τ in t+1:T, i in 1:batch_sizes[τ]
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
                (sim_batches[τ-t]["power"][i] / 2) * (
                    sum(
                        x_next[τ,i,j]
                        for j in DC.power_tilegroups_map[p]
                    )
                    + sum(
                        x_next[τ,i,j]
                        for j in intersect(DC.power_tilegroups_map[p], DC.power_tilegroups_map[p_])
                    )
                )
                for τ in t+1:T, i in 1:batch_sizes[τ]
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
                sim_batches[τ-t]["reward"][s,i] * y_next[τ,s,i,r]   
                for τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ], r in DC.row_IDs
            ) / S
        )
    elseif strategy in ["SSOA", "MPC"]
        @expression(
            model,
            future_assignment,
            sum(
                sim_batches[τ-t]["reward"][i] * y_next[τ,i,r]   
                for τ in t+1:T, i in 1:batch_sizes[τ], r in DC.row_IDs
            )
        )
    end

    if strategy == "myopic"
        @objective(model, Max, current_reward)
    elseif strategy in ["SSOA", "MPC", "SAA"]
        @objective(model, Max, current_reward + RCoeffsD.discount_factor * future_assignment)
    end

    verbose && println("Optimizing model...")
    
    optimize!(model)
    
    x_fixed_new = Dict{Tuple{Int, Int, Int}, Int}()
    y_fixed_new = Dict{Tuple{Int, Int, Int}, Int}()
    u_fixed_new = Dict{Tuple{Int, Int}, Int}()
    for i in 1:batch_sizes[t], j in DC.tilegroup_IDs
        val = round(JuMP.value(x_now[i,j]))
        if val > 0
            x_fixed_new[(t,i,j)] = val
        end
    end
    for i in 1:batch_sizes[t], r in DC.row_IDs
        val = round(JuMP.value(y_now[i,r]))
        if val > 0
            y_fixed_new[(t,i,r)] = val
            u_fixed_new[(t,i)] = val
        end
    end

    results = Dict(
        "x" => x_fixed_new,
        "y" => y_fixed_new,
        "u" => u_fixed_new,
        "time_taken" => time() - start_time,
        "objective" => JuMP.objective_value(model),
        "demands_placed" => sum(values(y_fixed_new)),
        "racks_placed" => sum(values(x_fixed_new)),
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
    if strategy in ["SAA"]
        results["x_next"] = Dict{Tuple{Int, Int, Int, Int}, Int}()
        for τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ], j in DC.tilegroup_IDs
            val = round(JuMP.value(x_next[τ,s,i,j]))
            if val > 0
                results["x_next"][(τ,s,i,j)] = val
            end
        end
        results["y_next"] = Dict{Tuple{Int, Int, Int, Int}, Int}()
        results["u_next"] = Dict{Tuple{Int, Int, Int}, Int}()
        for τ in t+1:T, s in 1:S, i in 1:batch_sizes[τ]
            val = round(JuMP.value(u_next[τ,s,i]))
            if val > 0
                results["u_next"][(τ,s,i)] = val
            end
            for r in DC.row_IDs
                val = round(JuMP.value(y_next[τ,s,i,r]))
                if val > 0
                    results["y_next"][(τ,s,i,r)] = val
                end
            end
        end
    elseif strategy in ["SSOA", "MPC"]
        results["x_next"] = Dict{Tuple{Int, Int, Int}, Int}()
        for τ in t+1:T, i in 1:batch_sizes[τ], j in DC.tilegroup_IDs
            val = round(JuMP.value(x_next[τ,i,j]))
            if val > 0
                results["x_next"][(τ,i,j)] = val
            end
        end
        results["y_next"] = Dict{Tuple{Int, Int, Int}, Int}()
        results["u_next"] = Dict{Tuple{Int, Int}, Int}()
        for τ in t+1:T, i in 1:batch_sizes[τ]
            val = round(JuMP.value(u_next[τ,i]))
            if val > 0
                results["u_next"][(τ,i)] = val
            end
            for r in DC.row_IDs
                val = round(JuMP.value(y_next[τ,i,r]))
                if val > 0
                    results["y_next"][(τ,i,r)] = val
                end
            end
        end
    end
    if with_precedence && t < T && strategy in ["SAA", "SSOA", "MPC"]
        if strategy in ["SAA"]
            results["precedence_now"] = Dict(
                (s, i, i_) => JuMP.value(precedence_now[s, i, i_])
                for s in 1:S, i in 1:batch_sizes[t], i_ in 1:batch_sizes[t+1]
            )
            results["precedence_next"] = Dict(
                (τ, s, i, i_) => JuMP.value(precedence_next[τ, s, i, i_])
                for τ in t+1:T-1, s in 1:S
                    for i in 1:batch_sizes[τ], i_ in 1:batch_sizes[τ+1]
            )
        elseif strategy in ["SSOA", "MPC"]
            results["precedence_now"] = Dict(
                (i, i_) => JuMP.value(precedence_now[i, i_])
                for i in 1:batch_sizes[t], i_ in 1:batch_sizes[t+1]
            )
            results["precedence_next"] = Dict(
                (τ, i, i_) => JuMP.value(precedence_next[τ, i, i_])
                for τ in t+1:T-1
                    for i in 1:batch_sizes[τ], i_ in 1:batch_sizes[τ+1]
            )
        end
    end
    if obj_minimize_rooms
        results["w"] = JuMP.value.(w_now)
        results["room_penalty"] = JuMP.value(room_penalty)
    end
    if obj_minimize_rows
        results["z"] = JuMP.value.(z_now)
        results["row_penalty"] = JuMP.value(row_penalty)
    end
    if obj_minimize_tilegroups
        results["v"] = JuMP.value.(v_now)
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
    batches::Vector{<:Dict{String, <:Array}},
    batch_sizes::Vector{Int}, 
    ;
    all_sim_batches::Union{Vector{<:Vector{<:Dict{String, <:Array}}}, Nothing} = nothing,
    env::Union{Gurobi.Env, Nothing} = nothing,
    strategy::String = "SSOA",
    S::Int = 1, # Number of sample paths
    seed::Union{Int, Nothing} = nothing,
    with_precedence::Bool = false,
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
    MIPGap::Float64 = 1e-4,
    time_limit_sec = 0,
    time_limit_sec_per_iteration = 60,
    verbose::Bool = true,
    test_run::Bool = false,
)
    if isnothing(env)
        env = Gurobi.Env()
    end
    if isnothing(seed)
        seed = abs(Random.rand(Int))
    end
    Random.seed!(seed)
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
    u_fixed = Dict{Tuple{Int, Int}, Int}()
    all_results = Dict{String, Any}[]

    for t in 1:T
        verbose && println("Starting iteration $t of $T:")

        # Simulate
        if strategy in ["SSOA", "SAA", "MPC"] && t < T
            if isnothing(all_sim_batches)
                sim_batches = simulate_batches(
                    strategy, Sim, 
                    RCoeffsD.placement_reward,
                    RCoeffsD.placement_var_reward,
                    t, T,
                    batch_sizes,
                    ;
                    S = S,
                )
                verbose && println("Simulated batches.")
            else
                sim_batches = all_sim_batches[t]
                verbose && println("Retrieved batches.")
            end
        else
            sim_batches = nothing
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
            ),
            verbose = verbose,
        )
        merge!(x_fixed, results["x"])
        merge!(y_fixed, results["y"])
        merge!(u_fixed, results["u"])
        if verbose
            println("--------------------------------")
            println("Iteration $t of $T completed in $(results["time_taken"]) s.")
            println("Placed $(length(results["x"])) new demands ($(length(x_fixed)) total).")
            println("Current assignment:  $(results["current_assignment"])")
            if strategy in ["SSOA", "SAA", "MPC"]
                println("Future assignment:   $(results["future_assignment"])")
            end
            if obj_minimize_rooms
                @printf("Room penalty:          %.2f\n", results["room_penalty"])
            end
            if obj_minimize_rows
                @printf("Row penalty:           %.2f\n", results["row_penalty"])
            end
            if obj_minimize_tilegroups
                @printf("Tilegroup penalty:     %.2f\n", results["tilegroup_penalty"])
            end
            if obj_minimize_power_surplus
                @printf("Power surplus penalty: %.2f\n", results["power_surplus_penalty"])
            end
            if obj_minimize_power_balance
                @printf("Power balance penalty: %.2f\n", results["power_balance_penalty"])
            end
            println("--------------------------------")
        end
        
        # Compute metrics for current iteration
        update_metrics!(DC, results)
        push!(all_results, deepcopy(results))

        # Update dynamic parameters
        update_dynamic_parameters!(RCoeffsD, DC, results)
        if test_run
            break
        end
    end

    return Dict(
        "x" => x_fixed,
        "y" => y_fixed,
        "u" => u_fixed,
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
            results["row_space_utilizations"][r] <= 0 ? (2.0 * RCoeffsD.row_mult) : (
                results["row_space_utilizations"][r] <= 10 ? (1.0 * RCoeffsD.row_mult) : 0.0
            )
        )
        for r in DC.row_IDs
    )
    RCoeffsD.room_penalties = Dict(
        m => (
            results["room_space_utilizations"][m] <= 0 ? (40.0 * RCoeffsD.room_mult) : (
                results["room_space_utilizations"][m] <= 0.3 * (
                    length(DC.room_rows_map[m]) * 20 # Number of tiles in room m
                ) ? (3.0 * RCoeffsD.room_mult) : 0.0
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
    batch_sizes::Vector{Int},
    DC::DataCenter,
    strategy::String,
    ;
    with_precedence::Bool = false,
    obj_minimize_rooms::Bool = true,
    obj_minimize_rows::Bool = true,
    obj_minimize_tilegroups::Bool = true,
    obj_minimize_power_surplus::Bool = true,
    obj_minimize_power_balance::Bool = true,
)
    T = length(all_results)
    keys_totake = [
        "time_taken", "optimality_gap",
        "objective",
        "demands_placed",
        "racks_placed",
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
        Dict(k => [all_results[t][k] for t in 1:T]
        for k in keys_totake)
    )
    room_space_utilization_data = DataFrame(Dict(
        "$m" => [
            all_results[t]["room_space_utilizations"][m]
            for t in 1:T
        ]
        for m in DC.room_IDs
    ))
    toppower_utilization_data = DataFrame(Dict(
        "$p" => [
            all_results[t]["toppower_utilizations"][p]
            for t in 1:T
        ]
        for p in DC.toppower_IDs
    ))
    result = Dict(
        "iteration_data" => iteration_data,
        "room_space_utilization_data" => room_space_utilization_data,
        "toppower_utilization_data" => toppower_utilization_data,
        "toppower_utilization" => (
            toppower_utilization_data[end, :] |> sum
        ) / (
            sum(DC.power_capacity[p] for p in DC.toppower_IDs)
        ),
    )
    result["demands_placed"] = sum(iteration_data[!, "demands_placed"])
    result["racks_placed"] = sum(iteration_data[!, "racks_placed"])
    result["objective"] = sum(iteration_data[!, "current_assignment"])
    if with_precedence
        first_ind_drop = T
        for t in 1:T
            if length(all_results[t]["y"]) < batch_sizes[t]
                first_ind_drop = t
                break
            end
        end
        # Values until (and including)the first drop
        result["demands_placed_precedence"] = sum(iteration_data[1:first_ind_drop, "demands_placed"])
        result["racks_placed_precedence"] = sum(iteration_data[1:first_ind_drop, "racks_placed"])
        result["objective_precedence"] = sum(iteration_data[1:first_ind_drop, "current_assignment"])
    end

    if obj_minimize_power_surplus || obj_minimize_power_balance
        result["toppower_pair_utilization_data"] = DataFrame(Dict(
            "$(p1), $(p2)" => [
                all_results[t]["toppower_pair_utilization"][(m, p1, p2)] / DC.power_balanced_capacity[m]
                for t in 1:length(all_results)
            ]
            for m in DC.room_IDs
                for (p1, p2) in Tuple.(collect(combinations(DC.room_toppower_map[m], 2)))
        ))
    end
    return result
end