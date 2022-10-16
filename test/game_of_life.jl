using StochasticAD
using Test
using Statistics

include("../tutorials/game_of_life/core.jl")

@testset "AD and Finite Differences" begin
    p = 0.5
    nsamples = 100_000
    samples_fd_clever = [GoLCore.fd_clever(p) for i in 1:nsamples]
    samples_st = [derivative_estimate(GoLCore.play, p) for i in 1:nsamples]

    @test mean(samples_st)≈mean(samples_fd_clever) rtol=5e-2
end
