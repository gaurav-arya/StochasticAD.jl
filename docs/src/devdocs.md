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
import Random # hide
Random.seed!(1234) # hide
using Distributions
function mygeometric(p)
    x = 0
    while !(rand(Bernoulli(p)))
        x += 1
    end
    return x
end
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
function mygeometric(p_st::StochasticTriple{T}) where {T}
    p = p_st.value
    x = mygeometric(p)

    # Form the new discrete perturbations (combinations of weight w and perturbation Y - X)
    Δs1 = if p_st.δ > 0
        # right stochastic derivative
        w = x / (p * (1 - p))
        x > 0 ? similar_new(p_st.Δs, -1, w) : similar_empty(p_st.Δs, Int)
    elseif p_st.δ < 0
        # left stochastic derivative
        w = (x + 1) / p # positive since the negativity of p_st.δ cancels out the negativity of w_L
        similar_new(p_st.Δs, 1, w)
    else
        similar_empty(p_st.Δs, Int)
    end

    # Propagate any existing perturbations to p through the function
    function map_func(Δ)
        # Sample mygeometric(p + Δ) independently. (A better strategy would be to couple to the original sample.)
        mygeometric(p + Δ) - x 
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
f(x) = mygeometric(x + 0.4 * rand(Bernoulli(x)))
@show stochastic_triple(f, 0.1)
nothing # hide
```