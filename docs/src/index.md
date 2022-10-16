# StochasticAD

`StochasticAD` is an experimental, research package for automatic differentiation (AD) of stochastic programs. It implements AD algorithms for handling functions which are *discrete* and *random*, based on the methodology developed in [TODO].

## Preview

Derivatives are all about how functions are affected by tiny changes in their input. To understand the effect of a tiny change, instead of providing a standard real number to a function, we can provide an object called a *stochastic triple*. First, let's consider a deterministic function.

```@example continuous
using StochasticAD
f(p) = p^2
stochastic_triple(f, 2) # Feeds 2 + ε into f
```
The output tells us that if we change the input `2` by a tiny amount `ε`, the output of `f` will change by around `4ε`. [This is the case we're familiar with](https://en.wikipedia.org/wiki/Dual_number). But what happens with a discrete random function? Let's give it a try. 
```@example discrete
import Random # hide
Random.seed!(4321) # hide
using StochasticAD, Distributions
f(p) = rand(Bernoulli(p)) # 1 with probability p, 0 otherwise
stochastic_triple(f, 0.5) # Feeds 0.5 + ε into f
```
The output of a [Bernoulli variable](https://en.wikipedia.org/wiki/Bernoulli_distribution) cannot change by a tiny amount: it is either `0` or `1`. But in the probabilistic world, there is another way to change by a tiny amount *on average*: jump by a large amount, with tiny probability. The purpose of the *third* component of the stochastic triple is to describe these perturbations. Here, the stochastic triple says that the original random output was `0`, but given a small change `ε` in the input, the output will jump up to `1` with probability around `2ε`.

Stochastic triples can be used to construct a new random program whose average is the derivative of the average of the original program. Let's try a crazier example, where we mix discrete and continuous randomness!
```@example estimate
using StochasticAD, Distributions
import Random # hide
Random.seed!(1234) # hide

function X(p)
    a = p^2 
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