#text # Simple reverse mode example 

##cell
#text Load our packages

using StochasticAD
using Distributions
using Enzyme
using LinearAlgebra

##cell
#text Let us define our target function

# Define a toy `StochasticAD`-differentiable function for computing an integer value from a string.
string_value(strings, index) = Int(sum(codepoint, strings[index]))
string_value(strings, index::StochasticTriple) = StochasticAD.propagate(index -> string_value(strings, index), index)

function f(θ; derivative_coupling = StochasticAD.InversionMethodDerivativeCoupling())
    strings = ["cat", "dog", "meow", "woofs"]
    index = randst(Categorical(θ / sum(θ)); derivative_coupling)
    return string_value(strings, index) * sum(θ)
end

θ = [0.1, 0.5, 0.3, 0.1]
@show f(θ)
nothing

##cell
# First, let's compute the sensitivity of `f` in a particular direction via forward-mode Stochastic AD.
u = [1.0, 2.0, 4.0, -7.0]
@assert iszero(sum(u))
@show derivative_estimate(f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
nothing

##cell
#text Uniform pruning with original func.

@show derivative_estimate(f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))

##cell

forward() = derivative_estimate(f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
reverse() = derivative_estimate(f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))

directional_derivs_fwd = [forward() for i in 1:10000]
derivs_bwd = [reverse() for i in 1:10000]
directional_derivs_bwd = [dot(u, δ) for δ in derivs_bwd]
println("Forward mode: $(mean(directional_derivs_fwd)) ± $(std(directional_derivs_fwd) / 100)")
println("Reverse mode: $(mean(directional_derivs_bwd)) ± $(std(directional_derivs_bwd) / 100)")

nothing

##cell
#! format: off #src
using Literate #src

function preprocess(content) #src
    new_lines = map(split(content, "\n")) do line #src
        if endswith(line, "#src") #src
            line #src
        elseif startswith(line, "##cell") #src
            "#src" #src
        elseif startswith(line, "#text") #src
            replace(line, "#text" => "#") #src
        # try and save comments; strip necessasry since Literate.jl also treats indented comments on their own line as markdown. #src
        elseif startswith(strip(line), "#") && !startswith(strip(line), "#=") && !startswith(strip(line), "#-") #src
            # TODO: should be replace first occurence only? #src
            replace(line, "#" => "##") #src
        else #src
            line #src
        end #src
    end #src
    return join(new_lines, "\n") #src
end #src

withenv("JULIA_DEBUG" => "Literate") do #src
    @time Literate.markdown(@__FILE__, joinpath(pwd(), "TODO"); execute = true, flavor = Literate.CommonMarkFlavor(), preprocess = preprocess) #src
end #src