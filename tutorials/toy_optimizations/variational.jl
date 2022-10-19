# Toy variational problem: Find Poisson(p) close to NegativeBinomial(10, 1-30/(10+30))
# by minimization of the Kullback Leibler distance
cd(@__DIR__)
using StochasticAD, Distributions, Optimisers
import Random # hide
Random.seed!(1234) # hide
PLOT = true
if PLOT
    using CairoMakie
end

# Sample the likelihood ratio. E[X(p)] is the Kullback-Leibler distance between the models
function X(p)
    i = rand(Poisson(p))
    return logpdf(Poisson(p), i) - logpdf(NegativeBinomial(10, 1 - 30 / (10 + 30)), i)
end

# Minimize E[X] = KL(Poisson(p)| NegativeBinomial(10, 1-30/(10+30))) using Adam and Optimize.jl
iterations = 5000
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

if PLOT
    dp = 1 / 2
    N = 1000
    ps = 10:dp:50
    avg = [mean(X(p) for _ in 1:N) for p in ps]
    derivative = [mean(derivative_estimate(X, p) for _ in 1:N) for p in ps]
    f = Figure()
    ax = f[1, 1] = Axis(f, title = "Estimates")
    lines!(ax, ps, avg, label = "≈ E[X(p)]")
    lines!(ax, ps, derivative, label = "≈ d/dp E[X(p)]")
    vlines!(ax, [p_opt], label = "p_opt", color = :green, linewidth = 2.0)
    hlines!(ax, [0.0], color = :black, linewidth = 1.0)

    f[1, 2] = Legend(f, ax, framevisible = false)
    ylims!(ax, (-10, 10))
    ax = f[2, 1:2] = Axis(f, title = "Optimizer trace")
    lines!(ax, trace, color = :green, linewidth = 2.0)
    save("variational.png", f)
    display(f)
end
