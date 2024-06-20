```@meta
EditURL = "../../../tutorials/reverse_example/reverse_demo.jl"
```

# Simple reverse mode example

```@setup random_walk
import Pkg
Pkg.activate("../../../tutorials")
Pkg.develop(path="../../..")
Pkg.instantiate()

````@example reverse_demo
#
````

import Random
Random.seed!(1234)
```

Load our packages

````@example reverse_demo
using StochasticAD
using Distributions
using Enzyme
using LinearAlgebra
````

Let us define our target function.

````@example reverse_demo
# Define a toy `StochasticAD`-differentiable function for computing an integer value from a string.
string_value(strings, index) = Int(sum(codepoint, strings[index]))
string_value(strings, index::StochasticTriple) = StochasticAD.propagate(index -> string_value(strings, index), index)

function f(θ; derivative_coupling = StochasticAD.InversionMethodDerivativeCoupling())
    strings = ["cat", "dog", "meow", "woofs"]
    index = randst(Categorical(θ); derivative_coupling)
    return string_value(strings, index)
end

θ = [0.1, 0.5, 0.3, 0.1]
@show f(θ)
nothing
````

First, let's compute the sensitivity of `f` in a particular direction via forward-mode Stochastic AD.

````@example reverse_demo
u = [1.0, 2.0, 4.0, -7.0]
@show derivative_estimate(f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
nothing
````

Now, let's do the same with reverse-mode.

````@example reverse_demo
@show derivative_estimate(f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))
````

Let's verify that our reverse-mode gradient is consistent with our forward-mode directional derivative.

````@example reverse_demo
forward() = derivative_estimate(f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
reverse() = derivative_estimate(f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))

N = 40000
directional_derivs_fwd = [forward() for i in 1:N]
derivs_bwd = [reverse() for i in 1:N]
directional_derivs_bwd = [dot(u, δ) for δ in derivs_bwd]
println("Forward mode: $(mean(directional_derivs_fwd)) ± $(std(directional_derivs_fwd) / sqrt(N))")
println("Reverse mode: $(mean(directional_derivs_bwd)) ± $(std(directional_derivs_bwd) / sqrt(N))")
@assert isapprox(mean(directional_derivs_fwd), mean(directional_derivs_bwd), rtol = 3e-2)

nothing
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

