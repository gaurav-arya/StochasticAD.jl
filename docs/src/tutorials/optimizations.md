# Stochastic optimization with discreteness

```@setup random_walk
import Pkg
Pkg.activate("../../../tutorials/toy_optimizations")
Pkg.develop(path="../../..")
Pkg.instantiate()
```

In this tutorial, we solve two stochastic optimization problems using `StochasticAD` where the objective contains discrete randomness. We will need the following packages:
```@example optimizations
using Distributions # defines several supported discrete distributions 
using StochasticAD
using CairoMakie # for plotting
using Optimisers # for stochastic gradient descent
```

## Optimizing our toy program

Recall the "crazy" program from the intro:
```@example optimizations
function X(p)
    a = p * (1 - p)
    b = rand(Binomial(10, p))
    c = 2 * b + 3 * rand(Bernoulli(p))
    return a * c * rand(Normal(b, a))
end
```

Let's maximize $\mathbb{E}[X(p)]$! First, let's setup the problem, using the [`StochasticModel`](@ref)helper utility to create a trainable model:
```@example optimizations
p0 = [0.5] # initial value of p
m = StochasticAD.StochasticModel(p0, x -> -X(x)) # Formulate as minimization problem
```
Now, let's perform stochastic gradient descent using [Adam](https://arxiv.org/abs/1412.6980).
```@example optimizations
# Minimize E[X] = KL(Poisson(p)| NegativeBinomial(10, 1-30/(10+30)))
iterations = 1000
trace = Float64[]
o = Adam() # use Adam for optimization
s = Optimisers.setup(o, m)
for i in 1:iterations
    # Perform a gradient step
    Optimisers.update!(s, m, StochasticAD.stochastic_gradient(m))
    push!(trace, m.p[])
end
p_opt = m.p[] # Our optimized value of p
```
Finally, let's plot the results of our optimization, and also perform a sweep through the parameter space to verify the accuracy of our estimate:
```@example optimizations
## Sweep through parameters to find average and derivative
ps = 0.02:0.02:0.98 # values of p to sweep
N = 1000 # number of samples at each p
avg = [mean(X(p) for _ in 1:N) for p in ps]
derivative = [mean(derivative_estimate(X, p) for _ in 1:N) for p in ps]

## Make plots
f = Figure()
ax = f[1, 1] = Axis(f, title = "Estimates", xlabel="Value of p")
lines!(ax, ps, avg, label = "≈ E[X(p)]")
lines!(ax, ps, derivative, label = "≈ d/dp E[X(p)]")
vlines!(ax, [p_opt], label = "p_opt", color = :green, linewidth = 2.0)
hlines!(ax, [0.0], color = :black, linewidth = 1.0)

f[1, 2] = Legend(f, ax, framevisible = false)
ylims!(ax, (-50, 80))
ax = f[2, 1:2] = Axis(f, title = "Optimizer trace", xlabel="Iterations", ylabel="Value of p")
lines!(ax, trace, color = :green, linewidth = 2.0)
save("crazy_opt.png", f,  px_per_unit = 4) # hide
```
![](crazy_opt.png)

## Solving a variational problem

Let's consider a toy variational program: we find a[Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) that is close to that of a [negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution), via minimization of the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $D_{\mathrm{KL}}$. Concretly, let us solve
```math
\underset{p \in \mathbb{R}}{\operatorname{argmin}} D_{\mathrm{KL}}\left(\mathrm{Pois}(p) \middle\| \mathrm{NB}(10, 0.25) \right).
```
The following program produces an unbiased estimate of the objective:
```@example optimizations
function X(p)
    i = rand(Poisson(p))
    return logpdf(Poisson(p), i) - logpdf(NegativeBinomial(10, 1 - 30 / (10 + 30)), i)
end
```
We can now optimize the KL divergence via stochastic gradient descent!
```@example optimizations
iterations = 1000
p0 = [10.0]
m = StochasticAD.StochasticModel(p0, X) # Formulate as minimization problem
trace = Float64[]
o = Adam(0.1)
s = Optimisers.setup(o, m)
for i in 1:iterations
    Optimisers.update!(s, m, StochasticAD.stochastic_gradient(m))
    push!(trace, m.p[])
end
p_opt = m.p[]
```
Let's plot our results in the same way as before:
```@example optimizations
ps = 10:0.5:50
N = 1000
avg = [mean(X(p) for _ in 1:N) for p in ps]
derivative = [mean(derivative_estimate(X, p) for _ in 1:N) for p in ps]
f = Figure()
ax = f[1, 1] = Axis(f, title = "Estimates", xlabel="Value of p")
lines!(ax, ps, avg, label = "≈ E[X(p)]")
lines!(ax, ps, derivative, label = "≈ d/dp E[X(p)]")
vlines!(ax, [p_opt], label = "p_opt", color = :green, linewidth = 2.0)
hlines!(ax, [0.0], color = :black, linewidth = 1.0)

f[1, 2] = Legend(f, ax, framevisible = false)
ylims!(ax, (-10, 10))
ax = f[2, 1:2] = Axis(f, title = "Optimizer trace", ylabel="Value of p", xlabel="Iterations")
lines!(ax, trace, color = :green, linewidth = 2.0)
save("variational.png", f, px_per_unit = 4) # hide
```
![](variational.png)
