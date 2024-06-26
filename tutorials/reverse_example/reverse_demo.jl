#text # Simple reverse mode example 

#text ```@setup random_walk
#text import Pkg
#text Pkg.activate("../../../tutorials")
#text Pkg.develop(path="../../..")
#text Pkg.instantiate()
#text
#text import Random 
#text Random.seed!(1234)
#text ```

import Random #src
Random.seed!(1234) #src

##cell
#text Load our packages

using StochasticAD
using Distributions
using Enzyme
using LinearAlgebra

##cell
#text Let us define our target function.

# Define a toy `StochasticAD`-differentiable function for computing an integer value from a string.
string_value(strings, index) = Int(sum(codepoint, strings[index]))
function string_value(strings, index::StochasticTriple)
    StochasticAD.propagate(index -> string_value(strings, index), index)
end

function f(θ; derivative_coupling = StochasticAD.InversionMethodDerivativeCoupling())
    strings = ["cat", "dog", "meow", "woofs"]
    index = randst(Categorical(θ); derivative_coupling)
    return string_value(strings, index)
end

θ = [0.1, 0.5, 0.3, 0.1]
@show f(θ)
nothing

##cell
#text First, let's compute the sensitivity of `f` in a particular direction via forward-mode Stochastic AD.
u = [1.0, 2.0, 4.0, -7.0]
@show derivative_estimate(
    f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
nothing

##cell
#text Now, let's do the same with reverse-mode, via [`EnzymeReverseAlgorithm`](@ref).

@show derivative_estimate(
    f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))

##cell
#text Let's verify that our reverse-mode gradient is consistent with our forward-mode directional derivative.

function forward()
    derivative_estimate(
        f, θ, StochasticAD.ForwardAlgorithm(PrunedFIsBackend()); direction = u)
end
function reverse()
    derivative_estimate(
        f, θ, StochasticAD.EnzymeReverseAlgorithm(PrunedFIsBackend(Val(:wins))))
end

N = 40000
directional_derivs_fwd = [forward() for i in 1:N]
derivs_bwd = [reverse() for i in 1:N]
directional_derivs_bwd = [dot(u, δ) for δ in derivs_bwd]
println("Forward mode: $(mean(directional_derivs_fwd)) ± $(std(directional_derivs_fwd) / sqrt(N))")
println("Reverse mode: $(mean(directional_derivs_bwd)) ± $(std(directional_derivs_bwd) / sqrt(N))")
@assert isapprox(mean(directional_derivs_fwd), mean(directional_derivs_bwd), rtol = 3e-2)

nothing

##cell
#! format: off #src
using Literate #src
do_documenter = true #src

function preprocess(content) #src
    new_lines = map(split(content, "\n")) do line #src
        if endswith(line, "#src") #src
            line #src
        elseif startswith(line, "##cell") #src
            "#src" #src
        elseif startswith(line, "#text") #src
            replace(line, "#text" => "#") #src
        # try and save comments; strip necessasry since Literate.jl also treats indented comments on their own line as markdown. #src
        elseif startswith(strip(line), "#") && !startswith(strip(line), "#=") &&
               !startswith(strip(line), "#-") #src
            # TODO: should be replace first occurence only? #src
            replace(line, "#" => "##") #src
        else #src
            line #src
        end #src
    end #src
    return join(new_lines, "\n") #src
end #src

withenv("JULIA_DEBUG" => "Literate") do #src
    dir = joinpath(dirname(dirname(pathof(StochasticAD))), "docs", "src", "tutorials") #src
    if do_documenter #src
        @time Literate.markdown(
            @__FILE__, dir; execute = false, flavor = Literate.DocumenterFlavor(),
            preprocess = preprocess, documenter = true) #src
    else #src
        @time Literate.markdown(@__FILE__, dir; execute = true,
            flavor = Literate.CommonMark(), preprocess = preprocess) #src
    end #src
end #src
