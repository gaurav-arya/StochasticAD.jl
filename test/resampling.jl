using StochasticAD
using Random, Test
using Distributions
using LinearAlgebra
using ForwardDiff

# test forward-mode AD and reverse-mode AD on the particle filter

### Particle Filter Functions
include("../tutorials/particle_filter/core.jl")
seed = 237347578

### Define model
Random.seed!(seed)

T = 3
d = 2
A(θ, a = 0.01) = [exp(-a)*cos(θ[]) exp(-a)*sin(θ[])
    -exp(-a)*sin(θ[]) exp(-a)*cos(θ[])]
obs(x, θ) = MvNormal(x, 0.01 * collect(I(d)))
dyn(x, θ) = MvNormal(A(θ) * x, 0.02 * collect(I(d)))
x0 = [2.0, 0.0] # start value of the simulation
start(θ) = Dirac(x0)
θtrue = [0.20]
# put it all together
stochastic_model = ParticleFilterCore.StochasticModel(T, start, dyn, obs)

### simulate model
Random.seed!(seed)
xs, ys = ParticleFilterCore.simulate_single(stochastic_model, θtrue)
###

### initialize sampler
m = 1000
particle_filter = ParticleFilterCore.ParticleFilter(m, stochastic_model, ys,
    ParticleFilterCore.sample_stratified)
###

@testset "new weight" begin
    p = 0.5
    st = stochastic_triple(p)
    d = ForwardDiff.Dual(p, (1.0, 2.0))
    @test new_weight(p) == one(p)
    @test StochasticAD.value(new_weight(st)) == one(p)
    @test StochasticAD.delta(new_weight(st)) == 1.0 / p
    @test ForwardDiff.value(new_weight(d)) == one(p)
    @test collect(ForwardDiff.partials(new_weight(d))) == [1.0 / p, 2.0 / p]
end

@testset "forward-mode and reverse-mode AD: single run" begin
    Random.seed!(seed)
    grad_forw = ParticleFilterCore.forw_grad(θtrue, particle_filter)
    Random.seed!(seed)
    grad_back = ParticleFilterCore.back_grad(θtrue, particle_filter)
    @test grad_forw ≈ grad_back
end

@testset "AD and Finite Differences" begin
    h = 0.02 # finite diff
    N = 500 # number of samples
    grad_fw = [ParticleFilterCore.forw_grad(θtrue, particle_filter)[1] for i in 1:N]
    # grad_bw = @time [back_grad(θtrue, particle_filter) for i in 1:N]
    grad_fd = [(ParticleFilterCore.log_likelihood(particle_filter, θtrue .+ h) -
                ParticleFilterCore.log_likelihood(particle_filter, θtrue .- h)) / (2h)
               for i in 1:N]

    @test mean(grad_fd)≈mean(grad_fw) rtol=5e-2
end
