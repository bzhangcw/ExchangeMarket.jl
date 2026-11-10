#!/usr/bin/env julia
# Same defaults and reporting as test-gpu.jl, running on the CPU.
ENV["LINEAR_DEVICE"] = "cpu"
include(joinpath(@__DIR__, "test_linear.jl"))
args = copy(ARGS)
any(a -> a == "--sizes_n" || startswith(a, "--sizes_n="), args) || append!(args, ["--sizes_n", "500"])
any(a -> a == "--sizes_m" || startswith(a, "--sizes_m="), args) || append!(args, ["--sizes_m", "1000"])
if abspath(PROGRAM_FILE) == @__FILE__
    LinearComparison.main(args)
end
