module StochasticADEnzymeExt

using StochasticAD
using Enzyme

function enzyme_target(u, X, p, backend)
    # equivalent to derivative_estimate(X, p; backend, direction = u), but specialize to real output to make Enzyme happier
    st = StochasticAD.stochastic_triple_direction(X, p, u; backend)
    if !(StochasticAD.valtype(st) <: Real)
        error("EnzymeReverseAlgorithm only supports real-valued outputs.")
    end
    return derivative_contribution(st)
end

function StochasticAD.derivative_estimate(X, p, alg::StochasticAD.EnzymeReverseAlgorithm;
        direction = nothing, alg_data = (; forward_u = nothing))
    if !isnothing(direction)
        error("EnzymeReverseAlgorithm does not support keyword argument `direction`")
    end
    if p isa AbstractVector
        Δu = zeros(float(eltype(p)), length(p))
        u = isnothing(alg_data.forward_u) ?
            rand(StochasticAD.RNG, float(eltype(p)), length(p)) : alg_data.forward_u
        autodiff(Enzyme.Reverse, enzyme_target, Active, Duplicated(u, Δu),
            Const(X), Const(p), Const(alg.backend))
        return Δu
    elseif p isa Real
        u = isnothing(alg_data.forward_u) ? rand(StochasticAD.RNG, float(typeof(p))) :
            forward_u
        ((du, _, _, _),) = autodiff(Enzyme.Reverse, enzyme_target, Active, Active(u),
            Const(X), Const(p), Const(alg.backend))
        return du
    else
        error("EnzymeReverseAlgorithm only supports p::Real or p::AbstractVector")
    end
end

end
