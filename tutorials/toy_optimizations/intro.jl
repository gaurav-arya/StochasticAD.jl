# Toy expectation optimization problem 
cd(@__DIR__)
using StochasticAD, Distributions, Optimisers
import Random # hide
Random.seed!(1234) # hide
PLOT = true
if PLOT
    using GLMakie
end

# The "crazy" stochastic program from the introduction
function X(p)
    a = p * (1 - p)
    b = rand(Binomial(10, p))
    c = 2 * b + 3 * rand(Bernoulli(p))
    return a * c * rand(Normal(b, a))
end

# Maximize E[X(p)] using Adam and Optimize
p0 = [0.5]
iterations = 5000
m = StochasticAD.StochasticModel(p0, x -> -X(x)) # Formulate as minimization problem
trace = Float64[]
o = Adam()
s = Optimisers.setup(o, m)
for i in 1:iterations
    Optimisers.update!(s, m, StochasticAD.stochastic_gradient(m))
    push!(trace, m.p[])
end
p_opt = m.p[]

if PLOT
    dp = 1 / 50
    N = 1000
    ps = dp:dp:(1 - dp)
    expected = [mean(X(p) for _ in 1:N) for p in ps]
    slope = [mean(derivative_estimate(X, p) for _ in 1:N) for p in ps]

    f = Figure()
    ax = f[1, 1] = Axis(f, title = "Estimates")
    lines!(ax, ps, expected, label = "≈ E X(p)")
    lines!(ax, ps, slope, label = "≈ (E X(p))'")
    vlines!(ax, [p_opt], label = "p_opt", color = :green, linewidth = 2.0)
    hlines!(ax, [0.0], color = :black, linewidth = 1.0)

    f[1, 2] = Legend(f, ax, framevisible = false)
    ylims!(ax, (-50, 80))
    ax = f[2, 1:2] = Axis(f, title = "Optimizer trace")
    lines!(ax, trace, color = :green, linewidth = 2.0)
    save("intro.png", f)
    display(f)
end
