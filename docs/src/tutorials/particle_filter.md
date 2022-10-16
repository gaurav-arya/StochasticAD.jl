# Differentiable particle filter

Using a bootstrap particle sampler, we can approximate the posterior distributions
of the states given noisy and partial observations of the state of a hidden Markov
model by a cloud of `K` weighted particles with weights `W`.

In this tutorial, we are going to:
- implement a differentiable particle filter based on `StochasticAD.jl`.
- visualize the particle filter in ``d = 2`` dimensions.
- compare the gradient based on the differentiable particle filter to a biased
  gradient estimator as well as to the gradient of a differentiable Kalman filter.
- show how to benchmark primal evaluation, forward- and reverse-mode AD of the
  particle filter.

## Setup

We will make use of several julia packages. For example, we are going to use
`Distributions` and `DistributionsAD` that implement the reparameterization trick
for Gaussian distributions used in the observation and state-transition model, which
we specify below. We also import `GaussianDistributions.jl` to implement the
differentiable Kalman filter.

### Package dependencies

```@example particle_filter
# activate tutorial project file
import Pkg # hide
Pkg.activate("../../../tutorials") # hide
Pkg.develop(path="../../..") # hide
Pkg.instantiate() # hide

# load dependencies
using StochasticAD
using Distributions
using DistributionsAD
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Zygote
using ForwardDiff
using GaussianDistributions
using GaussianDistributions: correct, ⊕
using Measurements
using UnPack
using Plots
using LaTeXStrings
using BenchmarkTools
```

### Particle filter

For convenience, we first introduce the new type `StochasticModel` with the following
fields:

- `T`: total number of time steps.
- `start`: starting distribution for the initial state. For example, in the form of a narrow
   Gaussian `start(θ) = Gaussian(x0, 0.001 * I(d))`.
- `dyn`: pointwise differentiable stochastic program in the form of Markov transition densities.
   For example, `dyn(x, θ) = MvNormal(reshape(θ, d, d) * x, Q(θ))`, where `Q(θ)` denotes the
   covariance matrix.
- `obs`: observation model having a smooth conditional probability density depending on
   current state `x` and parameters `θ`. For example, `obs(x, θ) = MvNormal(x, R(θ))`,
   where `R(θ)` denotes the covariance matrix.

For parameters `θ`,  `rand(start(θ))` gives a sample from the prior distribution of the
starting distribution. For current state `x` and parameters `θ`, `xnew = rand(dyn(x, θ))`
samples the new state (i.e. `dyn` gives for each `x, θ` a distribution-like object). Finally,
`y = rand(obs(x, θ))` samples an observation.

We can then define the `ParticleFilter` type that wraps a stochastic model `StochM::StochasticModel`,
a sampling strategy (with arguments `p, K, sump=1`) and observational data `ys`.
For simplicity, our implementation assumes a observation-likelihood function being available
via `pdf(obs(x, θ), y)`.

```@example particle_filter
struct StochasticModel{TType<:Integer,T1,T2,T3}
    T::TType # time steps
    start::T1 # prior
    dyn::T2 # dynamical model
    obs::T3 # observation model
end

struct ParticleFilter{mType<:Integer,MType<:StochasticModel,yType,sType}
    m::mType # number of particles
    StochM::MType # stochastic model
    ys::yType # observations
    sample_strategy::sType # sampling function
end
```

### Kalman filter

We consider a stochastic program that fulfills the assumptions of a Kalman filter.
We follow [Kalman.jl](https://github.com/mschauer/Kalman.jl/blob/master/README.md) to implement a differentiable version.
Our `KalmanFilter` type wraps a stochastic model `StochM::StochasticModel` and observational data `ys`. It assumes a
observation-likelihood function is implemented via `llikelihood(yres, S)`. The Kalman filter
contains the following fields:

- `d`: dimension of the state-transition matrix ``\Phi`` according to ``x = \Phi x + w`` with ``w \sim \operatorname{Normal}(0,Q)``.
- `StochM`: Stochastic model of type `StochasticModel`.
- `H`: linear map from the state space into the observed space according to ``y = H x + \nu`` with ``\nu \sim \operatorname{Normal}(0,R)``.
- `R`: covariance matrix entering the observation model according to ``y = H x + \nu`` with ``\nu \sim \operatorname{Normal}(0,R)``.
- `Q`: covariance matrix entering the state-transition model according to ``x = \Phi x + w`` with ``w \sim \operatorname{Normal}(0,Q)``.
- `ys`: observations.


```@example particle_filter
llikelihood(yres, S) = GaussianDistributions.logpdf(Gaussian(zero(yres), Symmetric(S)), yres)
struct KalmanFilter{dType<:Integer,MType<:StochasticModel,HType,RType,QType,yType}
    # H, R = obs
    # θ, Q = dyn
    d::dType
    StochM::MType # stochastic model
    H::HType # observation model, maps the true state space into the observed space
    R::RType # observation model, covariance matrix
    Q::QType # dynamical model, covariance matrix
    ys::yType # observations
end
```

To get observations `ys` from the latent states `xs` based on the
(true, potentially unknown) parameters `θ`, we simulate a single particle
from the forward model returning a vector of observations (no resampling steps).

```@example particle_filter
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
```

A particle filter becomes efficient if resampling steps are included. Resampling
is numerically attractive because particles with small weight are discarded, so
computational resources are not wasted on particles with vanishing weight.

Here, let us implement a stratified resampling strategy, see for example
[Murray (2012)](https://arxiv.org/abs/1202.6163), where `p` denotes the probabilities of `K` particles
with `sump = sum(p)`.

```@example particle_filter
function sample_stratified(p, K, sump=1)
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
```

This sampling strategy can be used within a differentiable resampling step in our
particle filter using the `use_new_weight` function as implemented in
`StochasticAD.jl`. The `resample` function below returns the states `X_new`
and weights `W_new` of the resampled particles.

- `m`: number of particles.
- `X`: current particle states.
- `W`: current weight vector of the particles.
- `ω == sum(W)` is an invariant.
- `sample_strategy`: specific resampling strategy to be used. For example, `sample_stratified`.
- `use_new_weight=true`: Allows one to switch between biased, stop-gradient method and
   differentiable resampling step.

```@example particle_filter
function resample(m, X, W, ω, sample_strategy, use_new_weight=true)
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
```

Note that we added a `if` condition that allows us to switch between the differentiable
resampling step and the stop-gradient approach.

We're now equipped with all primitive operations to set up the particle filter,
which propagates particles with weights `W` preserving the invariant `ω == sum(W)`.
We never normalize `W` and, therefore, `ω` in the code below contains likelihood
information. The particle-filter implementation defaults to return particle
positions and weights at `T` if `store_path=false` and takes the following input
arguments:

- `θ`: parameters for the stochastic program (state-transition and observation model).
- `store_path=false`: Option to store the path of the particles, e.g. to visualize/inspect
  their trajectories.
- `use_new_weight=true`: Option to switch between the stop-gradient and our differentiable
  resampling step method. Defaults to using differentiable resampling.
- `s`: controls the number of resampling steps according to `t > 1 && t < T && (t % s == 0)`.


```@example particle_filter
function (F::ParticleFilter)(θ; store_path=false, use_new_weight=true, s=1)
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
```

Following [Kalman.jl](https://github.com/mschauer/Kalman.jl/blob/master/README.md), we implement
a differentiable Kalman filter to check the ground-truth gradient. Our Kalman filter
returns an updated posterior state estimate and the log-likelihood and takes the
parameters of the stochastic program as an input.

```@example particle_filter
function (F::KalmanFilter)(θ)
    @unpack d, StochM, H, R, Q = F
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
```

For both filters, it is straightforward to obtain the log-likelihood via:

```@example particle_filter
function log_likelihood(F::ParticleFilter, θ, use_new_weight=true, s=1)
    _, W = F(θ; store_path=false, use_new_weight=use_new_weight, s=s)
    log(sum(W))
end
```
and
```@example particle_filter
function log_likelihood(F::KalmanFilter, θ)
    _, ll = F(θ)
    ll
end
```

For convenience, we define functions for
- forward-mode AD (and differentiable resampling step) to compute the gradient of
  the log-likelihood of the particle filter.
- reverse-mode AD (and differentiable resampling step) to compute the gradient of
  the log-likelihood of the particle filter.
- forward-mode AD (and stop-gradient method) to compute the gradient of
  the log-likelihood of the particle filter (without the `new_weight` function).
- forward-mode AD to compute the gradient of the log-likelihood of the Kalman filter.

```@example particle_filter

forw_grad(θ, F::ParticleFilter; s=1) = ForwardDiff.gradient(θ -> log_likelihood(F, θ, true, s), θ)
back_grad(θ, F::ParticleFilter; s=1) = Zygote.gradient(θ -> log_likelihood(F, θ, true, s), θ)[1]
forw_grad_biased(θ, F::ParticleFilter; s=1) = ForwardDiff.gradient(θ -> log_likelihood(F, θ, false, s), θ)
forw_grad_Kalman(θ, F::KalmanFilter) = ForwardDiff.gradient(θ -> log_likelihood(F, θ), θ)
```

## Model

Having set up all core functionalities, we can now define the specific stochastic
model.

We consider the following system with a ``d``-dimensional latent process,

```math
\begin{aligned}
x_i &= \Phi x_{i-1} + w_i &\text{ with } w_i \sim \operatorname{Normal}(0,Q),\\
y_i &= x_i + \nu_i &\text{ with } \nu_i \sim \operatorname{Normal}(0,R),
\end{aligned}
```

where ``\Phi`` is a ``d``-dimensional rotation matrix.

```@example particle_filter
seed = 423897

### Define model
# here: n-dimensional rotation matrix
Random.seed!(seed)
T = 20 # time steps
d = 2 # dimension
# generate a rotation matrix
M = randn(d, d)
c = 0.3 # scaling
O = exp(c * (M - transpose(M)) / 2)
@assert det(O) ≈ 1
@assert transpose(O) * O ≈ I(d)
θtrue = vec(O) # true parameter

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
stochastic_model = StochasticModel(T, start, dyn, obs)

# relevant corresponding Kalman filterng defs
H_Kalman = collect(I(d))
R_Kalman = Gaussian(zeros(Float64, d), R)
# Φ_Kalman = O
Q_Kalman = Gaussian(zeros(Float64, d), Q)
###

### simulate model
Random.seed!(seed)
xs, ys = simulate_single(stochastic_model, θtrue)
```

## Visualization

Using `particle_filter(θ; store_path=true)` and `kalman_filter(θ)`, it is
straightforward to visualize both filters for our observed data.

```@example particle_filter
m = 1000
kalman_filter = KalmanFilter(d, stochastic_model, H_Kalman, R_Kalman, Q_Kalman, ys)
particle_filter = ParticleFilter(m, stochastic_model, ys, sample_stratified)
```


```@example particle_filter
### run and visualize filters
Xs, W = particle_filter(θtrue; store_path=true)
fig = plot(getindex.(xs, 1), getindex.(xs, 2), legend=false, xlabel=L"x_1", ylabel=L"x_2") # x1 and x2 are bad names..conflictng notation
scatter!(fig, getindex.(ys, 1), getindex.(ys, 2))
for i in 1:min(m, 100) # note that Xs has obs noise.
    local xs = [Xs[t][i] for t in 1:T]
    scatter!(fig, getindex.(xs, 1), getindex.(xs, 2), marker_z=1:T, color=:cool, alpha=0.1) # color to indicate time step
end

xs_Kalman, ll_Kalman = kalman_filter(θtrue)
plot!(getindex.(mean.(xs_Kalman), 1), getindex.(mean.(xs_Kalman), 2), legend=false, color="red")
display(fig) # hide
png("pf_1") # hide
```
![](pf_1.png)

## Bias

We can also investigate the distribution of the gradients from the particle filter
with and without differentiable resampling step, as compared to the gradient computed
by differentiating the Kalman filter.

```@example particle_filter
### compute gradients
Random.seed!(seed)
X = [forw_grad(θtrue, particle_filter) for i in 1:200] # gradient of the particle filter *with* differentiation of the resampling step
Random.seed!(seed)
Xbiased = [forw_grad_biased(θtrue, particle_filter) for i in 1:200] # Gradient of the particle filter *without* differentiation of the resampling step
# pick an arbitrary coordinate
index = 1 # take derivative with respect to first parameter (2-dimensional example has a rotation matrix with four parameters in total)
# plot histograms for the sampled derivative values
fig = plot(normalize(fit(Histogram, getindex.(X, index), nbins=20), mode=:pdf), legend=false) # ours
plot!(normalize(fit(Histogram, getindex.(Xbiased, index), nbins=20), mode=:pdf)) # biased
vline!([mean(X)[index]], color=1)
vline!([mean(Xbiased)[index]], color=2)
# add derivative of differentiable Kalman filter as a comparison
XK = forw_grad_Kalman(θtrue, kalman_filter)
vline!([XK[index]], color="black")
display(fig) # hide
png("pf_2") # hide
```
![](pf_2.png)

The estimator using the `new_weight` function agrees with the gradient value from
the Kalman filter and the [particle filter AD scheme developed by Ścibior and Wood](https://arxiv.org/abs/2106.10314),
unlike biased estimators that neglect the contribution of the derivative from the
resampling step. However, the biased estimator displays a smaller variance.

## Benchmark

Finally, we can use `BenchmarkTools.jl` to benchmark the run times of the primal
pass with respect to forward-mode and reverse-mode AD of the particle filter. As
expected, forward-mode AD outperforms reverse-mode AD for the small number of
parameters considered here.

```@example particle_filter
# secs for how long the benchmark should run, see https://juliaci.github.io/BenchmarkTools.jl/stable/
secs = 2

suite = BenchmarkGroup()
suite["scaling"] = BenchmarkGroup(["grads"])

suite["scaling"]["primal"] = @benchmarkable log_likelihood(particle_filter, θtrue)
suite["scaling"]["forward"] = @benchmarkable forw_grad(θtrue, particle_filter)
suite["scaling"]["backward"] = @benchmarkable back_grad(θtrue, particle_filter)

tune!(suite)
results = run(suite, verbose=true, seconds=secs)

t1 = measurement(mean(results["scaling"]["primal"].times), std(results["scaling"]["primal"].times) / sqrt(length(results["scaling"]["primal"].times)))
t2 = measurement(mean(results["scaling"]["forward"].times), std(results["scaling"]["forward"].times) / sqrt(length(results["scaling"]["forward"].times)))
t3 = measurement(mean(results["scaling"]["backward"].times), std(results["scaling"]["backward"].times) / sqrt(length(results["scaling"]["backward"].times)))
@show t1 t2 t3

ts = (t1, t2, t3) ./ 10^6 # ms
@show ts
```
