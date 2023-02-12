module RandomWalkCore

using Random
using Statistics
using Distributions
using LinearAlgebra
using StochasticAD
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

function simulate_walk(probs, steps, n; debug = false, hardcode_leftright_step = false)
    X = 0
    for i in 1:n
        probs_X = probs(X) # transition probabilities
        debug && @show probs_X
        step_index = rand(Categorical(probs_X)) # produces an integer-valued StochasticTriple
        debug && @show step_index
        if hardcode_leftright_step
            step = 2 * (step_index - 1) - 1
        else
            step = steps[step_index] # differentiate through array indexing
        end
        X += step
        debug && @show X
    end
    return X
end

X(p, n; kwargs...) = simulate_walk(make_probs(p), steps, n; kwargs...)
fX(p, n; kwargs...) = f(X(p, n; kwargs...))
X(p; kwargs...) = X(p, n; kwargs...)
fX(p; kwargs...) = fX(p, n; kwargs...)

## Simulate with score method manually added on

function simulate_walk_score(probs, steps, n; debug = false)
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

## Exactly compute transition matrix M

range = 0:n
range_start = 1 # range[range_start] = 0

function get_M(p)
    probs = make_probs(p)
    M = zeros(eltype(first(probs(range[range_start]))), length(range), length(range))
    low = minimum(range)
    for x in range
        for (step, prob) in zip(steps, probs(x))
            if (x + step) in range
                M[x + step - low + 1, x - low + 1] = prob
            end
        end
    end
    M
end

function probdensity(p, n)
    M = get_M(p)
    vec = zeros(length(range))
    vec[range_start] = 1
    M^n * vec
end

get_dX(p, n) = sum(probdensity(p, n) .* range)
get_dfX(p, n) = sum(probdensity(p, n) .* (f.(range)))

end
