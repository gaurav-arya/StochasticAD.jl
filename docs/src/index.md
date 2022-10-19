```@raw html
<img class="display-light-only" src="images/path_skeleton.png">
<img class="display-dark-only" src="images/path_skeleton_dark.png">
```

# StochasticAD

[StochasticAD](https://github.com/gaurav-arya/StochasticAD.jl) is an experimental, research package for automatic differentiation (AD) of stochastic programs.
It implements AD algorithms for handling programs that can contain *discrete* randomness, based on the methodology developed in [this NeurIPS 2022 paper](https://doi.org/10.48550/arXiv.2210.08572).

## Introduction

Derivatives are all about how functions are affected by a tiny change `ε` in their input. First, let's imagine perturbing the input of a deterministic, differentiable function such as $f(p) = p^2$ at $p = 2$.
```@example continuous
using StochasticAD
f(p) = p^2
stochastic_triple(f, 2) # Feeds 2 + ε into f
```
The output tells us that if we change the input `2` by a tiny amount `ε`, the output of `f` will change by approximately `4ε`. This is the case we're familiar with: we can get the value `4` by applying the chain rule, $\frac{\mathrm{d}}{\mathrm{d} p} p^2 = 2p = 4$. Thinking in terms of tiny changes, the output above looks a lot like a [dual number](https://en.wikipedia.org/wiki/Dual_number). But what happens with a discrete random function? Let's give it a try. 
```@example discrete
import Random # hide
Random.seed!(4321) # hide
using StochasticAD, Distributions
f(p) = rand(Bernoulli(p)) # 1 with probability p, 0 otherwise
stochastic_triple(f, 0.5) # Feeds 0.5 + ε into f
```
The output of a [Bernoulli variable](https://en.wikipedia.org/wiki/Bernoulli_distribution) cannot change by a tiny amount: it is either `0` or `1`. But in the probabilistic world, there is another way to change by a tiny amount *on average*: jump by a large amount, with tiny probability. `StochasticAD` introduces a stochastic triple object, which generalizes dual numbers by including a *third* component to describe these perturbations. Here, the stochastic triple says that the original random output was `0`, but given a small change `ε` in the input, the output will jump up to `1` with probability approximately `2ε`.

Stochastic triples can be used to construct a new random program whose average is the derivative of the average of the original program. We simply propagate stochastic triples through the program via [`stochastic_triple`](@ref), and then sum up the "dual" and "triple" components at the end via [`derivative_contribution`](@ref). This process is packaged together in the function [`derivative_estimate`](@ref). Let's try a crazier example, where we mix discrete and continuous randomness!
```@example estimate
using StochasticAD, Distributions
import Random # hide
Random.seed!(1234) # hide

function X(p)
    a = p * (1 - p)
    b = rand(Binomial(10, p))
    c = 2 * b + 3 * rand(Bernoulli(p))
    return a * c * rand(Normal(b, a))
end

st = @show stochastic_triple(X, 0.6) # sample a single stochastic triple at p = 0.6
@show derivative_contribution(st) # which produces a single derivative estimate...

samples = [derivative_estimate(X, 0.6) for i in 1:1000] # many samples from derivative program
derivative = mean(samples)
uncertainty = std(samples) / sqrt(1000)
println("derivative of 𝔼[X(p)] = $derivative ± $uncertainty")
```

## Index

See the [public API](public_api.md) for a walkthrough of the API, and the tutorials on differentiating a [random walk](tutorials/random_walk.md), a [stochastic game of life](tutorials/game_of_life.md), and a [particle filter](tutorials/particle_filter.md), and solving [stochastic optimization and variational inference problems](tutorials/optimizations.md) with discrete randomness. This is a prototype package with a number of [limitations](limitations.md).

