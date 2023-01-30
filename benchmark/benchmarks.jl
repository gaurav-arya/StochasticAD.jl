using BenchmarkTools

include("random_walk.jl")

const SUITE = BenchmarkGroup()
SUITE["random_walk"] = RandomWalkBenchmark.suite
