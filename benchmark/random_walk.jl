module RandomWalkBenchmark

using BenchmarkTools

using StochasticAD
using Statistics
using ForwardDiff: derivative
include("../tutorials/random_walk/core.jl")
using .RandomWalkCore: n, p, nsamples
using .RandomWalkCore: fX, get_dfX

const suite = BenchmarkGroup()

suite["original"] = @benchmarkable $(fX)($p)
suite["PrunedFIs"] = @benchmarkable derivative_estimate($fX, $p;
                                                        backend = StochasticAD.PrunedFIs)
suite["PrunedFIsAggressive"] = @benchmarkable derivative_estimate($fX, $p;
                                                                  backend = StochasticAD.PrunedFIsAggressive)

end
