using Parameters
@with_kw struct RackPlacementCoefficients
    placement_reward::Float64 = 200.0
    discount_factor::Float64 = 0.1
    # alpha in paper 
    balance::Float64 = 1e-5
    # beta in paper
    balance_exceeded::Float64 = 1e-3
end