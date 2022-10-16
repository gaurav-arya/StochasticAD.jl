module RandomWalkCore

using Random
using Statistics
using Distributions
using LinearAlgebra
using StochasticAD
using BenchmarkTools
using StaticArrays
using OffsetArrays: Origin
import ForwardDiff
using ForwardDiff: Dual, derivative, value, partials

## Parameters

steps = SA[-1, 1]
make_probs(p) = X -> SA[1 - exp(-X / p), exp(-X / p)]
f = x -> x^2 # function to apply to X

n = 50# number of steps
p = 100 # default parameter value
n_range = 10:10:100 # range for testing asymptotics
p_range = 2 .* n_range

nsamples = 10000 # number of times to run gradient estimators

## Simulate
function simulate_walk(probs, steps, n, debug = false)
    X = 0
    for i in 1:n
        probs_X = probs(X) # transition probabilities
        debug && @show probs_X
        step_index = rand(Categorical(probs_X)) # produces an integer-valued StochasticTriple
        debug && @show step_index
        step = steps[step_index] # differentiate through array indexing
        X += step
        debug && @show X
    end
    return X
end

X(p, n) = simulate_walk(make_probs(p), steps, n)
fX(p, n) = f(X(p, n))
X(p) = X(p, n)
fX(p) = fX(p, n)

## Simulate with score method manually added on
function simulate_walk_score(probs, steps, n, debug = false)
    X = 0.0
    dlogP = 0.0
    for i in 1:n
        probs_X = probs(X) # transition probabilities
        step_index = rand(Categorical(probs_X)) # just a number
        step = steps[step_index] # differentiate through array indexing
        dlogP += partials(log(probs_X[step_index]))[1]
        X += step # take step
    end
    return (X, dlogP)
end

score_X(p, n) = simulate_walk_score(make_probs(Dual(p, 1.0)), steps, n)
function score_X_deriv(p, n, avg)
    X, dlogP = score_X(p, n)
    (X - avg) * dlogP
end
function score_fX_deriv(p, n, avg)
    X, dlogP = score_X(p, n)
    return (f(X) - avg) * dlogP
end
score_X_deriv(p; avg = 0.0) = score_X_deriv(p, n, avg)
score_fX_deriv(p; avg = 0.0) = score_fX_deriv(p, n, avg)

end
