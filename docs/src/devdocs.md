# Developer documentation (WIP)

## Writing a custom rule for stochastic triples

### via `StochasticAD.propagate`

To handle a deterministic discrete construct that `StochasticAD` does not automatically handle (e.g. branching via `if`, boolean comparisons), it is often sufficient to simply add a dispatch rule that calls out to `StochasticAD.propagate`.

```@docs
StochasticAD.propagate
```

### via a custom dispatch

If a function does not meet the conditions of `StochasticAD.propagate` and is not already supported, a custom
dispatch may be necessary. For example, consider the following function which manually implements a geometric random variable:

```@example rule
import Random
Random.seed!(1234) # hide
using Distributions
# make rng input explicit
function mygeometric(rng, p)
    x = 0
    while !(rand(rng, Bernoulli(p)))
        x += 1
    end
    return x
end
mygeometric(p) = mygeometric(Random.default_rng(), p)
```

This is equivalent to `rand(Geometric(p))` which is already supported, but for pedagogical purposes we will
implement our own rule from scratch. Using the stochastic derivative formulas from [Automatic Differentiation of Programs with Discrete Randomness](https://doi.org/10.48550/arXiv.2210.08572), the right stochastic derivative of this program is given by
```math
Y_R = X - 1, w_R = \frac{x}{p(1-p)},
```
and the left stochastic derivative of this program is given by
```math
Y_L = X + 1, w_L = -\frac{x+1}{p}.
```

Using these expressions, we can now write the dispatch rule for stochastic triples:

```@example rule
using StochasticAD
import StochasticAD: StochasticTriple, similar_new, similar_empty, combine
function mygeometric(rng, p_st::StochasticTriple{T}) where {T}
    p = p_st.value
    rng_copy = copy(rng) # save a copy for coupling later
    x = mygeometric(rng, p)

    # Form the new discrete perturbations (combinations of weight w and perturbation Y - X)
    Δs1 = if p_st.δ > 0
        # right stochastic derivative
        w = p_st.δ * x / (p * (1 - p))
        x > 0 ? similar_new(p_st.Δs, -1, w) : similar_empty(p_st.Δs, Int)
    elseif p_st.δ < 0
        # left stochastic derivative
        w = -p_st.δ * (x + 1) / p # positive since the negativity of p_st.δ cancels out the negativity of w_L
        similar_new(p_st.Δs, 1, w)
    else
        similar_empty(p_st.Δs, Int)
    end

    # Propagate any existing perturbations to p through the function
    function map_func(Δ)
        # Couple the samples by using the same RNG. (A simpler strategy would have been independent sampling, i.e. mygeometric(p + Δ) - x)
        mygeometric(copy(rng_copy), p + Δ) - x 
    end
    Δs2 = map(map_func, p_st.Δs)

    # Return the output stochastic triple
    StochasticTriple{T}(x, zero(x), combine((Δs2, Δs1)))
end
```
In the above, we used some of the interface functions supported by a collection of perturbations `Δs::StochasticAD.AbstractFIs`. These were `similar_empty(Δs, V)`, which created an empty perturbation of type `V`, `similar_new(Δs, Δ, w)`, which created a new perturbation of size `Δ` and weight `w`, `map(map_func, Δs)`,
which propagates a collection of perturbations through a mapping function, and `combine((Δs2, Δs1)))` which combines multiple collections of perturbations together.

We can test out our rule:
```@example rule
@show stochastic_triple(mygeometric, 0.1)

# try feeding an input that already has a pertrubation
f(x) = mygeometric(2 * x + 0.1 * rand(Bernoulli(x)))^2
@show stochastic_triple(f, 0.1)

# verify against black-box finite differences
N = 1000000
samples_stochad = [derivative_estimate(f, 0.1) for i in 1:N]
samples_fd = [(f(0.105) - f(0.095)) / 0.01 for i in 1:N]

println("Stochastic AD: $(mean(samples_stochad)) ± $(std(samples_stochad) / sqrt(N))")
println("Finite differences: $(mean(samples_fd)) ± $(std(samples_fd) / sqrt(N))")

nothing # hide
```

## Distribution-specific customization of differentiation algorithm 

```@docs
StochasticAD.randst
```