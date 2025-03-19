using Parameters
@with_kw struct RackPlacementCoefficients
    placement_reward::Float64 = 200.0
    discount_factor::Float64 = 0.1
    room_penalty::Float64 = 3.0 # 40, 3, 0 (at 0, 0.2, 1.0)
    row_penalty::Float64 = 1.0 # 2, 1, 0 (at 0, 0.5, 1.0) 
    tilegroup_penalty::Float64 = 1.0
    # alpha in paper 
    balance::Float64 = 1e-5
    # beta in paper
    balance_exceeded::Float64 = 1e-3
end