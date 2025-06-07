# Online Rack Placement

This repository documents code for the paper: "Online Rack Placement in Large-Scale Data Centers: Online Sampling Optimization and Deployment", found [here](https://arxiv.org/abs/2501.12725).

## Directory structure

Under `models/model.jl`, we define:
- `rack_placement_oracle()` which optimizes server rack placements with perfect information on future requests;
- `rack_placement()` which optimizes rack placements in an online fashion. The key parameter is the `strategy` parameter which can be:
    - `"myopic"`: this makes placement decisions in a myopic manner, only considering current requests
    - `"MPC"` this implements the certainty-equivalent (CE) heuristic (Algorithm 2), replacing uncertain future parameters by their mean, and resolving after each iteration.
    - `"SSOA"` and `"SAA"` both implement our OSO algorithm (Algorithm 1), sampling uncertain future parameters. They differ by the number of sample paths; `"SSOA"` uses 1 sample path and `"SAA"` uses `"S"` > 1 paths.

Examples can be found in `experiment.jl`.