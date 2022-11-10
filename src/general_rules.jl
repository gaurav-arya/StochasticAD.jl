"""
    define_triple_overload(sig)

Given a function signature, defines operator overloading rules for stochastic triples.
Currently supports functions with all-real inputs and one real output.
"""
# TODO: special case optimizations
# TODO: generalizations to not-all-real inputs and/or not-one-real output
function define_triple_overload(sig)
    opT, argTs = Iterators.peel(ExprTools.parameters(sig))
    opT <: Type{<:Type} && return  # not handling constructors
    sig <: Tuple{Type, Vararg{Any}} && return
    opT <: Core.Builtin && return false  # can't do operator overloading for builtins

    isabstracttype(opT) || fieldcount(opT) == 0 || return false  # not handling functors
    isempty(argTs) && return false  # we are an operator overloading AD, need operands
    all(argT isa Type && Real <: argT for argT in argTs) || return

    N = length(ExprTools.parameters(sig)) - 1  # skip the op

    if opT.instance in UNARY_PREDICATES
        for m in methods(opT.instance, (StochasticTriple,))
            if m.sig <: Tuple{opT, StochasticTriple}
                return
            end
        end
        @eval function (f::$opT)(st::StochasticTriple)
            val = value(st)
            out = f(val)
            if !alltrue(map(Δ -> (f(val + Δ) == out), st.Δs))
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return out
        end
    elseif opT.instance in BINARY_PREDICATES
        for m in methods(opT.instance, (StochasticTriple, StochasticTriple))
            if m.sig <: Tuple{opT, StochasticTriple, StochasticTriple}
                return
            end
        end
        # Special case equality comparisons as in https://github.com/JuliaDiff/ForwardDiff.jl/pull/481
        if opT.instance == Base.:(==)
            return_value_real = quote
                out && iszero(delta(st))
            end
            return_value_st = quote
                out2 = out && (delta(st1) == delta(st2))
            end
        else
            return_value_real = quote
                out
            end
            return_value_st = quote
                out
            end
        end
        @eval function (f::$opT)(st::StochasticTriple, x::Real)
            val = value(st)
            out = f(val, x)
            if !alltrue(map(Δ -> (f(val + Δ, x) == out), st.Δs))
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return $return_value_real
        end
        @eval function (f::$opT)(x::Real, st::StochasticTriple)
            val = value(st)
            out = f(x, val)
            if !alltrue(map(Δ -> (f(x, val + Δ) == out), st.Δs))
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return $return_value_real
        end
        @eval function (f::$opT)(st1::StochasticTriple, st2::StochasticTriple)
            val1 = value(st1)
            val2 = value(st2)
            out = f(val1, val2)

            safe_perturb1 = alltrue(map(Δ -> (f(val1 + Δ, val2) == out), st1.Δs))
            safe_perturb2 = alltrue(map(Δ -> (f(val1, val2 + Δ) == out), st2.Δs))
            if !safe_perturb1 || !safe_perturb2
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return $return_value_st
        end
    elseif N == 1
        if Base.return_types(frule, (Tuple{NoTangent, Real}, opT, Real))[1] <:
           Tuple{Any, NoTangent}
            return
        end
        # TODO: see if this is a compilation/precompilation bottleneck
        for m in methods(opT.instance, (StochasticTriple,))
            if m.sig <: Tuple{opT, StochasticTriple}
                return
            end
        end
        @eval function (f::$opT)(st::StochasticTriple{T}; kwargs...) where {T}
            args_tangent = (NoTangent(), delta(st))
            val, δ0 = frule(args_tangent, f, value(st); kwargs...)
            δ = (δ0 isa ZeroTangent || δ0 isa NoTangent) ? zero(value(st)) : δ0
            if !iszero(st.Δs)
                Δs = map(Δ -> f(st.value + Δ) - val, st.Δs)
            else
                Δs = similar_empty(st.Δs, typeof(val))
            end
            return StochasticTriple{T}(val, δ, Δs)
        end
    elseif N == 2
        if Base.return_types(frule, (Tuple{NoTangent, Real, Real}, opT, Real, Real))[1] <:
           Tuple{Any, NoTangent}
            return
        end
        for m in methods(opT.instance, (StochasticTriple, StochasticTriple))
            if m.sig <: Tuple{opT, StochasticTriple, StochasticTriple}
                return
            end
        end
        for R in AMBIGUOUS_TYPES
            @eval function (f::$opT)(st::StochasticTriple{T}, x::$R; kwargs...) where {T}
                args_tangent = (NoTangent(), delta(st), zero(x))
                val, δ0 = frule(args_tangent, f, value(st), x; kwargs...)
                δ = (δ0 isa ZeroTangent || δ0 isa NoTangent) ? zero(value(st)) : δ0
                if !iszero(st.Δs)
                    Δs = map(Δ -> f(st.value + Δ, x) - val, st.Δs)
                else
                    Δs = similar_empty(st.Δs, typeof(val))
                end
                return StochasticTriple{T}(val, δ, Δs)
            end
            @eval function (f::$opT)(x::$R, st::StochasticTriple{T}; kwargs...) where {T}
                args_tangent = (NoTangent(), zero(x), delta(st))
                val, δ0 = frule(args_tangent, f, x, value(st); kwargs...)
                δ = (δ0 isa ZeroTangent || δ0 isa NoTangent) ? zero(value(st)) : δ0
                if !iszero(st.Δs)
                    Δs = map(Δ -> f(x, st.value + Δ) - val, st.Δs)
                else
                    Δs = similar_empty(st.Δs, typeof(val))
                end
                return StochasticTriple{T}(val, δ, Δs)
            end
        end
        @eval function (f::$opT)(sts::Vararg{StochasticTriple{T}, 2}; kwargs...) where {T}
            args_tangent = (NoTangent(), delta.(sts)...)
            args = (f, value.(sts)...)
            val, δ0 = frule(args_tangent, args...; kwargs...)
            δ = (δ0 isa ZeroTangent || δ0 isa NoTangent) ? zero(value(st)) : δ0

            Δs_all = map(st -> getfield(st, :Δs), sts)
            if all(iszero.(Δs_all))
                Δs = similar_empty(first(sts).Δs, typeof(val))
            else
                Δs_coupled = couple(Tuple(Δs_all))
                vals_in = value.(sts)
                mapfunc = let vals_in = vals_in
                    Δ -> (f((vals_in .+ Δ)...) - val)
                end
                Δs = map(mapfunc, Δs_coupled)
            end
            return StochasticTriple{T}(val, δ, Δs)
        end
    end
end

on_new_rule(define_triple_overload, frule)

### Integer functions

"""
    Base.getindex(C::AbstractArray, st::StochasticTriple{T})

A simple prototype rule for array indexing. Assumes that underlying type of `st` can index into collection C.
"""
# TODO: support multiple indices, cartesian indices, non abstract array indexables, other use cases...
function Base.getindex(C::AbstractArray, st::StochasticTriple{T}) where {T}
    val = C[st.value]
    function do_map(Δ)
        value(C[st.value + Δ]) - value(val)
    end
    Δs = map(do_map, st.Δs)
    if val isa StochasticTriple
        Δs = combine((Δs, val.Δs))
    end
    return StochasticTriple{T}(value(val), delta(val), Δs)
end
