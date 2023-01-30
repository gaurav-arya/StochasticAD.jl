using StochasticAD
using Test
using Statistics

include("../tutorials/game_of_life/core.jl")
using .GoLCore: fd_clever, play, p, nsamples

@testset "AD and Finite Differences" begin
    samples_fd_clever = [fd_clever(p) for i in 1:nsamples]
    samples_st = [derivative_estimate(play, p) for i in 1:nsamples]

    @test mean(samples_st)≈mean(samples_fd_clever) rtol=5e-2
end
