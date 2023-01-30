using StochasticAD
using Test
using Statistics
using ForwardDiff: derivative

include("../tutorials/random_walk/core.jl")
using .RandomWalkCore: n, p, nsamples
using .RandomWalkCore: fX, get_dfX

@testset "Check unbiasedness" begin
    fX_deriv = derivative(p -> get_dfX(p, n), p)
    fX_deriv_estimate = mean(derivative_estimate(fX, p) for i in 1:nsamples)
    @test isapprox(fX_deriv, fX_deriv_estimate; rtol = 1e-2)
end
