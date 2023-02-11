include("core.jl")
println("## Exact computation\n")

using ForwardDiff: derivative
using BenchmarkTools
using .RandomWalkCore: n, p, nsamples
using .RandomWalkCore: X, f, fX, get_dX, get_dfX
using .RandomWalkCore: score_X_deriv, score_fX_deriv
using StochasticAD
using Statistics
import Random

X_deriv = derivative(p -> get_dX(p, n), p)
fX_deriv = derivative(p -> get_dfX(p, n), p)
println("X derivative: $X_deriv")
println("f(X) derivative: $fX_deriv")
println()

println("## Stochastic triple computation\n")

@btime fX(p)
@enter ForwardDiff.derivative(p -> fX(p; hardcode_leftright_step = true), p)
@btime derivative_estimate(fX, p; backend = StochasticAD.PrunedFIsAggressive)

triple_X_derivs = [derivative_estimate(X, p) for i in 1:nsamples]
triple_fX_derivs = [derivative_estimate(fX, p) for i in 1:nsamples]
println("Stochastic triple X derivative mean: $(mean(triple_X_derivs))")
println("Stochastic triple X derivative std : $(std(triple_X_derivs))")
println("Stochastic triple f(X) derivative mean: $(mean(triple_fX_derivs))")
println("Stochastic triple f(X) derivative std: $(std(triple_fX_derivs))")
println()

smoothed_X_derivs = [derivative(p -> X(p; hardcode_leftright_step = true), p)
                     for i in 1:nsamples]
smoothed_fX_derivs = [derivative(p -> fX(p; hardcode_leftright_step = true), p)
                      for i in 1:nsamples]
println("Smoothed X derivative mean: $(mean(smoothed_X_derivs))")
println("Smoothed X derivative std : $(std(smoothed_X_derivs))")
println("Smoothed f(X) derivative mean: $(mean(smoothed_fX_derivs))")
println("Smoothed f(X) derivative std: $(std(smoothed_fX_derivs))")
println()

println("## Score function computation\n")

# baseline
avg_X = mean(X(p) for i in 1:10000)
avg_fX = mean(fX(p) for i in 1:10000)
score_X_derivs = [score_X_deriv(p; avg = avg_X)
                  for i in 1:nsamples]
score_fX_derivs = [score_fX_deriv(p; avg = avg_fX)
                   for i in 1:nsamples]
println("Score X derivative mean: $(mean(score_X_derivs))")
println("Score X derivative std: $(std(score_X_derivs))")
println("Score f(X) derivative mean: $(mean(score_fX_derivs))")
println("Score f(X) derivative std: $(std(score_fX_derivs))")
println()

println("## Finite differences\n")

function fd(X, p, h = 10)
    state = copy(Random.default_rng())
    run1 = X(p - h / 2)
    copy!(Random.default_rng(), state)
    run2 = X(p + h / 2)
    (run2 - run1) / h
end

fd_X_derivs = [fd(X, p) for i in 1:nsamples]
fd_fX_derivs = [fd(f ∘ X, p) for i in 1:nsamples]
println("FD X derivative mean: $(mean(fd_X_derivs))")
println("FD X derivative std: $(std(fd_X_derivs))")
println("FD f(X) derivative mean: $(mean(fd_fX_derivs))")
println("FD f(X) derivative std: $(std(fd_fX_derivs))")
println()
