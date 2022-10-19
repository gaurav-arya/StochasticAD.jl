# Poisson autoregression
cd(@__DIR__)
using StochasticAD, Distributions
using Optimisers
import Random
Random.seed!(1234)
PLOT = true
if PLOT
    using GLMakie
end

# Poisson autoregression model accumulating a likelihood for starting λ
function igarch(a, b, c, n, λ) # todo: this should be a Turing/Tilde model
    ℓ = logpdf(Exponential(1.0), λ)
    z = rand(Poisson(λ))
    ℓ += logpdf(Poisson(λ), z)
    λ = a + b*z + c*λ
    for i in 2:n
        z = rand(Poisson(λ))
        ℓ += logpdf(Poisson(λ), z)
        λ = a + b*z + c*λ
    end
    return ℓ, λ, z
end

λ0 = 5.42 # true starting value

## Generate observations
#n = 10
#_, _, z_obs = igarch(a,b,c,n,λ0) # 140 in first run

# Likelihood of parameter p=λ0 given z_obs=140 (assume we don't know)
function X(p, z_obs = 140, n = 10)
    a, b, c = [0.25, 0.9, 0.51]
    ℓ, λ, _ = igarch(a,b,c,n,p)
    logpdf(Poisson(λ), z_obs)
end


# Maximize likelihood Adam and Optimize
p0 = [0.5]
iterations = 5000
m = StochasticAD.StochasticModel(p0, x->-X(x)) # Formulate as minimization problem
trace = Float64[]
o = Adam(0.02)
s = Optimisers.setup(o, m)
for i in 1:iterations
    Optimisers.update!(s, m, StochasticAD.stochastic_gradient(m))
    push!(trace, m.p[])
end
p_opt = m.p[]

if PLOT
    ps = range(0, 10, length=50)
    N = 1000
    expected = [mean(X(p) for _ in 1:N) for p in ps]
    slope = [mean(derivative_estimate(X, p) for _ in 1:N) for p in ps]

    f = Figure()
    ax = f[1, 1] = Axis(f, title="Estimates")
    lines!(ax, ps, expected, label="≈ E X(p)")
    lines!(ax, ps, slope, label="≈ (E X(p))'")
    vlines!(ax, [p_opt], label="p_opt", color=:green, linewidth=2.0)
    vlines!(ax, [λ0], linestyle=:dot, linewidth=2.0)
    hlines!(ax, [0.0], color=:black, linewidth=1.0)

    f[1, 2] = Legend(f, ax, framevisible = false)
    ylims!(ax, (-50,80))
    ax = f[2, 1:2] = Axis(f, title="Optimizer trace")
    lines!(ax, trace, color=:green, linewidth=2.0)
    hlines!(ax, [λ0], linestyle=:dot, linewidth=2.0)
    ylims!(ax, (0, 7))
    save("igarch.png", f)
    display(f)
end