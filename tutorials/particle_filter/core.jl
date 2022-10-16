module ParticleFilterCore

# load dependencies
using Distributions
using DistributionsAD
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Zygote
using StochasticAD
using ForwardDiff
using GaussianDistributions
using GaussianDistributions: correct, ⊕
using UnPack

### Particle Filter Functions

# Model defs

"""
    StochasticModel{dType<:Integer,TType<:Integer,T1,T2,T3}

For parameters `θ`,  `rand(start(θ))` gives a sample from the prior distribution of the
starting distribution. For current state `x` and parameters `θ`, `xnew = rand(dyn(x, θ))`
samples the new state (i.e. `dyn` gives for each `x, θ` a distribution-like object). Finally,
`y = rand(obs(x, θ))` samples an observation.

## Constructor

- `T`: total number of time steps.
- `start`: starting distribution for the initial state. For example, in the form of a narrow
   Gaussian `start(θ) = Gaussian(x0, 0.001 * I(d))`.
- `dyn`: pointwise differentiable stochastic program in the form of Markov transition densities.
   For example, `dyn(x, θ) = MvNormal(reshape(θ, d, d) * x, Q(θ))`, where `Q(θ)` denotes the
   covariance matrix.
- `obs`: observation model having a smooth conditional probability density depending on
   current state `x` and parameters `θ`. For example, `obs(x, θ) = MvNormal(x, R(θ))`,
   where `R(θ)` denotes the covariance matrix.
"""
struct StochasticModel{TType <: Integer, T1, T2, T3}
    T::TType # time steps
    start::T1 # prior
    dyn::T2 # dynamical model
    obs::T3 # observation model
end

# Particle filter
"""

    ParticleFilter{mType<:Integer,MType<:StochasticModel,yType,sType}

Wraps a stochastic model `StochM::StochasticModel` and observational data `ys`.
Assumes a observation-likelihood is available via `pdf(obs(x, θ), y)`.

## Constructor

- `m`: number of particles.
- `StochM`: stochastic model of type `StochasticModel`.
- `ys`: observations.
- `sample_strategy`: strategy for the resampling step of the particle filter. For example,
  stratified sampling as implemented in `sample_stratified`.
"""
struct ParticleFilter{mType <: Integer, MType <: StochasticModel, yType, sType}
    m::mType # number of particles
    StochM::MType # stochastic model
    ys::yType # observations
    sample_strategy::sType # sampling function
end

# Kalman filter
"""

    KalmanFilter{dType<:Integer,MType<:StochasticModel,HType,RType,QType,yType}

Differentiable Kalman filter following https://github.com/mschauer/Kalman.jl/blob/master/README.md.
Wraps a stochastic model `StochM::StochasticModel` and observational data `ys`. Assumes a
observation-likelihood is implemented via `llikelihood(yres, S)`. For example:
 ```
 llikelihood(yres, S) = GaussianDistributions.logpdf(Gaussian(zero(yres), Symmetric(S)), yres)
 ```

## Constructor

- `d`: dimension of the state-transition matrix Φ according to x = Φ*x + w with w ~ Normal(0,Q).
- `StochM`: Stochastic model of type `StochasticModel`.
- `H`: linear map from the state space into the observed space according to y = H x + ν with ν ~ Normal(0, R).
- `R`: covariance matrix entering the observation model according to y = H x + ν with ν ~ Normal(0, R).
- `Q`: covariance matrix entering the state-transition model according to x = Φ*x + w with w ~ Normal(0,Q).
- `ys`: observations.
"""
struct KalmanFilter{dType <: Integer, MType <: StochasticModel, HType, RType, QType, yType}
    # H, R = obs
    # θ, Q = dyn
    d::dType
    StochM::MType # stochastic model
    H::HType # observation model, maps the true state space into the observed space
    R::RType # observation model, covariance matrix
    Q::QType # dynamical model, covariance matrix
    ys::yType # observations
end

"""
    simulate_single(StochM::StochasticModel, θ)

Simulate a single particle from the forward model returning
a vector of observations (no resampling steps), e.g.
```
Random.seed!(seed)
xs, ys = simulate_single(StochM, θtrue)
```
to get observations ys from the latent states xs based on the
(true, potentially unknown) parameters θ.
"""
function simulate_single(StochM::StochasticModel, θ)
    @unpack T, start, dyn, obs = StochM
    x = rand(start(θ))
    y = rand(obs(x, θ))
    xs = [x]
    ys = [y]
    for t in 2:T
        x = rand(dyn(x, θ))
        y = rand(obs(x, θ))
        push!(xs, x)
        push!(ys, y)
    end
    xs, ys
end

"""
    sample_stratified(p, K, sump=1)

Stratified resampling strategy, see for example https://arxiv.org/abs/1202.6163.
Here, `p` denotes the probabilities of `K` particles with `sump = sum(p)`.
"""
function sample_stratified(p, K, sump = 1)
    n = length(p)
    U = rand()
    is = zeros(Int, K)
    i = 1
    cw = p[1]
    for k in 1:K
        t = sump * (k - 1 + U) / K
        while cw < t && i < n
            i += 1
            @inbounds cw += p[i]
        end
        is[k] = i
    end
    return is
end

"""
    resample(m, X, W, ω, sample_strategy, use_new_weight=true)

Resampling step wrapped for use in particle filter using differentiable
resampling from the article (`use_new_weight`). Returns states `X_new`
and weights `W_new` of resampled particles.

## args
- `m`: number of particles.
- `X`: current particle states.
- `W`: current weight vector of the particles.
- `ω == sum(W)` is an invariant.
- `sample_strategy`: specific resampling strategy to be used. Currently, only `sample_stratified` is available.
- `use_new_weight=true`: Allows one to switch between biased, stop-gradient method and
   differentiable resampling step.
"""
function resample(m, X, W, ω, sample_strategy, use_new_weight = true)
    js = Zygote.ignore(() -> sample_strategy(W, m, ω))
    X_new = X[js]
    if use_new_weight
        # differentiable resampling
        W_chosen = W[js]
        W_new = map(w -> ω * new_weight(w / ω) / m, W_chosen)
    else
        # stop gradient, biased approach
        W_new = fill(ω / m, m)
    end
    X_new, W_new
end

"""
 (F::ParticleFilter)(θ; store_path=false, use_new_weight=true, s=1)

Run particle filter. The particle filter propagates particles with weights `W` preserving the
invariant `ω == sum(W)`. `W` is never normalized and `ω` contains therefore likelihood information.
Defaults to return particle positions and weights at `T` if `store_path=false`.

## args
- `θ`: parameters for the stochastic program (state-transition and observation model).
- `store_path=false`: Option to store the path of the particles, e.g. to visualize/inspect
  their trajectories.
- `use_new_weight=true`: Option to switch between the stop-gradient and our differentiable
  resampling step method. Defaults to using differentiable resampling.
- `s`: controls the number of resampling steps according to `t > 1 && t < T && (t % s == 0)`.
"""
function (F::ParticleFilter)(θ; store_path = false, use_new_weight = true, s = 1)
    # s controls the number of resampling steps
    @unpack m, StochM, ys, sample_strategy = F
    @unpack T, start, dyn, obs = StochM

    X = [rand(start(θ)) for j in 1:m] # particles
    W = [1 / m for i in 1:m] # weights
    ω = 1 # total weight
    store_path && (Xs = [X])
    for (t, y) in zip(1:T, ys)
        # update weights & likelihood using observations
        wi = map(x -> pdf(obs(x, θ), y), X)
        W = W .* wi
        ω_old = ω
        ω = sum(W)
        # resample particles
        if t > 1 && t < T && (t % s == 0) # && 1 / sum((W / ω) .^ 2) < length(W) ÷ 32
            X, W = resample(m, X, W, ω, sample_strategy, use_new_weight)
        end
        # update particle states
        if t < T
            X = map(x -> rand(dyn(x, θ)), X)
            store_path && Zygote.ignore(() -> push!(Xs, X))
        end
    end
    (store_path ? Xs : X), W
end

# differentiable Kalman filter, following https://github.com/mschauer/Kalman.jl/blob/master/README.md
function llikelihood(yres, S)
    GaussianDistributions.logpdf(Gaussian(zero(yres), Symmetric(S)), yres)
end

"""
    (F::KalmanFilter)(θ)

Run differentiable Kalman filter. Returns updated posterior state estimate and log likelihood.

## args
- `θ`: parameters for the stochastic program (state-transition and observation model).
"""
function (F::KalmanFilter)(θ)
    @unpack d, StochM, H, R, Q, ys = F
    @unpack start = StochM

    x = start(θ)
    Φ = reshape(θ, d, d)

    x, yres, S = GaussianDistributions.correct(x, ys[1] + R, H)
    ll = llikelihood(yres, S)
    xs = Any[x]
    for i in 2:length(ys)
        x = Φ * x ⊕ Q
        x, yres, S = GaussianDistributions.correct(x, ys[i] + R, H)
        ll += llikelihood(yres, S)

        push!(xs, x)
    end
    xs, ll
end

# compute log-likelihood of Particle Sampler
"""
   log_likelihood(F::ParticleFilter, θ, use_new_weight=true, s=1)

Compute log-likelihood of particle sampler. See `ParticleFilter` for `use_new_weight` and `s`.

## args
- `θ`: parameters for the stochastic program (state-transition and observation model).
"""
function log_likelihood(F::ParticleFilter, θ, use_new_weight = true, s = 1)
    _, W = F(θ; store_path = false, use_new_weight = use_new_weight, s = s)
    log(sum(W))
end

# compute log-likelihood of Kalman Filter
"""
   log_likelihood(F::KalmanFilter, θ)

Compute log-likelihood of Kalman filter.

## args
- `θ`: parameters for the stochastic program (state-transition and observation model).
"""
function log_likelihood(F::KalmanFilter, θ)
    _, ll = F(θ)
    ll
end

# forward differentiation of particle sampler
function forw_grad(θ, F::ParticleFilter; s = 1)
    ForwardDiff.gradient(θ -> log_likelihood(F, θ, true, s), θ)
end
# backward differentiation of particle sampler
function back_grad(θ, F::ParticleFilter; s = 1)
    Zygote.gradient(θ -> log_likelihood(F, θ, true, s), θ)[1]
end
# biased forward differentiation of particle sampler, avoiding differentiation of the resampling step
function forw_grad_biased(θ, F::ParticleFilter; s = 1)
    ForwardDiff.gradient(θ -> log_likelihood(F, θ, false, s), θ)
end
# forward-mode AD of Kalman filter
forw_grad_Kalman(θ, F::KalmanFilter) = ForwardDiff.gradient(θ -> log_likelihood(F, θ), θ)
end
