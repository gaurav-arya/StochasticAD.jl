using BenchmarkTools

include("random_walk.jl")
include("game_of_life.jl")
include("iteration.jl")

const SUITE = BenchmarkGroup()
SUITE["random_walk"] = RandomWalkBenchmark.suite
SUITE["game_of_life"] = GoLBenchmark.suite
SUITE["iteration"] = IterationBenchmark.suite
