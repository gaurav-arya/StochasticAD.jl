# Random scattering (deflection) of a single particle

```@setup random_scattering
import Pkg
Pkg.activate("../../../tutorials")
Pkg.develop(path="../../..")
Pkg.instantiate()
```

## Intro and Setup
In this tutorial we implement and differentiate through a single particle scattering process.

In short, when a particle travels through some material with finite depth, at each step, the
particle has a chance (!) to interact, when it does, it loses energy and randomly changes direction
[^1].

```@example random_scattering
using StochasticAD, Statistics, Distributions
```

First of all, the interaction probability is controlled by a parameter `par`, think of this as
material property (~density), which we might be intereted in differentiate against to optimize
material.

```@example random_scattering

"""
Probability of interaction is a function of parameter we're intereted in diff. against
"""
function interact_prob(x,par)
    return 1/(1+exp(-(x-par)/0.2))*0.5
end
```

Then, we implement the "one step" in propagating particle, note instead of `if scattered
...`, we need to do it branchlessly, we also decide when interaction happens, we lose 10% of energy.
```@example random_scattering
"""
A single step of propagation, next1 and next2 are branchless "next state of particle"
"""
function propagate_scatter(E,phi,x,y,par)
    x = x + cos(phi)
    y = y + sin(phi)

    prob = interact_prob(x,par)
    scattered = rand(Bernoulli(prob))

    next1 = (E,x,y,phi)

    E0 = E*0.9 # lose 10% of energy
    phi0 = phi + 0.01*randn() # change direction
    next2 = (E0,x,y,phi0)
    
    # TODO: use better scattering coupling
    idx = 1+scattered
    return Tuple([next1[i], next2[i]][idx] for i in 1:4) # TODO: make this more ergonomic
end

```

Also, we need a stopping criteria, here we assume the material extends to `x=10` in space, thus when
particle leaves the volumn, we stop keeping it:
```@example random_scattering
decidetokeep(x, a, b) = x < 10 ? a : b
decidetokeep(x::StochasticAD.StochasticTriple, a, b) = StochasticAD.propagate(decidetokeep, x, a, b)
```

Finally, the main process:
```@example random_scattering
function generate_scatter(;E = 50.0, phi = 0.0, y = 0.0, par = 2.5)
    x = zero(par)
    xs = typeof(x)[]
    while true # TODO: improve while loop coupling (software + coupling strength)
	    next = propagate_scatter(E,phi,x,y,par)
        # "c" for candidate
		Ec, xc, yc, phic = next
        E, x, y, phi = decidetokeep(x, (Ec, xc, yc, phic), (E, x, y, phi))
        # TODO: make this automatic
        if StochasticAD.value(x) >= 10
            if !(x isa StochasticAD.StochasticTriple)
                break
            else
                xval = StochasticAD.value(x)
                if StochasticAD.alltrue(Δ -> (Δ + xval) >= 10, x.Δs)
                   break 
                end
            end
        end
        push!(xs, x)
	end
    return E
end
```

The logic behind `make this automatic` part is the following: after some amount of iteration, the
primal may have terminated but with alternative still going (or vise versa), so we need to check
that not only the primal has terminated, but all of the perterbations have also terminated.

```@example random_scattering
obj(par) = generate_scatter(; par)

@show obj(2.5)

@show derivative_estimate(obj, 2.5)
samples_st = [derivative_estimate(obj, 2.5) for i in 1:10000]
@show mean(samples_st)
@show std(samples_st) / sqrt(length(samples_st))
```

[^1]: Technically, energy loss of charged particle (dE/dx) happens continuously in dense material,
    so this "chance to lose energy" might be more like mean free path in gas. But we just want
    something simple to picture and implement.

