# ParticleFilter Model

using Random, LinearAlgebra, GaussianDistributions, Distributions

# particle filter core function definitions
include("core.jl")

### Define model

d = 2 # dimension
T = 20 # time steps

# generate a rotation matrix, dynamical model, observation model, prior distribution as a function of d
function generate_system(d, T)
    # here: n-dimensional rotation matrix
    seed = 423897
    Random.seed!(seed)

    M = randn(d, d)
    c = 0.3 # scaling
    O = exp(c * (M - transpose(M)) / 2)
    @assert det(O) ≈ 1
    @assert transpose(O) * O ≈ I(d)
    # true parameter
    θtrue = vec(O)

    # observation model
    R = 0.01 * collect(I(d))
    obs(x, θ) = MvNormal(x, R) # y = H x + ν with ν ~ Normal(0, R)

    # dynamical model
    Q = 0.02 * collect(I(d))
    dyn(x, θ) = MvNormal(reshape(θ, d, d) * x, Q) #  x = Φ*x + w with w ~ Normal(0,Q)

    # starting position
    x0 = randn(d)
    # prior distribution
    start(θ) = Gaussian(x0, 0.001 * collect(I(d)))

    # put it all together
    stochastic_model = ParticleFilterCore.StochasticModel(T, start, dyn, obs)

    # relevant corresponding Kalman filterng defs
    H_Kalman = collect(I(d))
    R_Kalman = Gaussian(zeros(Float64, d), R)
    # Φ_Kalman = O
    Q_Kalman = Gaussian(zeros(Float64, d), Q)

    ### simulate model
    Random.seed!(seed)
    xs, ys = ParticleFilterCore.simulate_single(stochastic_model, θtrue)
    ###

    ### initialize filters
    m = 1000 # number of particles
    kalman_filter = ParticleFilterCore.KalmanFilter(
        d, stochastic_model, H_Kalman, R_Kalman,
        Q_Kalman, ys)
    particle_filter = ParticleFilterCore.ParticleFilter(m, stochastic_model, ys,
        ParticleFilterCore.sample_stratified)

    return θtrue, xs, ys, stochastic_model, kalman_filter, particle_filter
end

θtrue, xs, ys, stochastic_model, kalman_filter, particle_filter = generate_system(d, T)
