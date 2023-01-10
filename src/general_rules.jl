"""
Operators which have already been overloaded by StochasticAD. 
"""
const handled_ops = Tuple{DataType, Int}[]

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

    if (opT, N) in handled_ops
        return
    end

    push!(handled_ops, (opT, N))

    if opT.instance in UNARY_PREDICATES && (N == 1)
        @eval function (f::$opT)(st::StochasticTriple)
            val = value(st)
            out = f(val)
            if !alltrue(map(Δ -> (f(val + Δ) == out), st.Δs))
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return out
        end
    elseif opT.instance in BINARY_PREDICATES && (N == 2)
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

            Δs_coupled = couple((st1.Δs, st2.Δs))
            safe_perturb = alltrue(map(Δs -> f(val1 + Δs[1], val2 + Δs[2]) == out,
                                       Δs_coupled))
            if !safe_perturb
                error("Output of boolean predicate cannot depend on input (unsupported by StochasticAD)")
            end
            return $return_value_st
        end
    elseif N == 1
        if Base.return_types(frule, (Tuple{NoTangent, Real}, opT, Real))[1] <:
           Tuple{Any, NoTangent}
            return
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

### Extra overloads

# TODO: generalize the below logic to compactly handle a wider range of functions.
# See also https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/dual.jl.

function Base.hash(st::StochasticTriple, hsh::UInt)
    if !isempty(st.Δs)
        error("Hashing a stochastic triple with perturbations not yet supported.")
    end
    hash(StochasticAD.value(st), hsh)
end

#=
This is a hacky experimental way to convert a float-like stochastic triple
into an integer-like one, to facilitate generic coding.
=#
function Base.round(I::Type{<:Integer}, st::StochasticTriple{T, V}) where {T, V}
    return StochasticTriple{T}(round(I, st.value), map(Δ -> round(I, st.value + Δ), st.Δs))
end

for op in UNARY_TYPEFUNCS_NOWRAP
    @eval function Base.$op(::Type{<:StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        return Base.$op(V)
    end
end

for op in UNARY_TYPEFUNCS_WRAP
    @eval function Base.$op(::Type{StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        return StochasticTriple{T, V, FIs}(Base.$op(V), zero(V), empty(FIs))
    end
    if !(op in (:(zero), :(one)))
        @eval function Base.$op(st::StochasticTriple)
            return Base.$op(typeof(st))
        end
    end
end

for op in RNG_TYPEFUNCS_WRAP
    @eval function Random.$op(rng::AbstractRNG,
                              ::Type{StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        return StochasticTriple{T, V, FIs}(Random.$op(rng, V), zero(V), empty(FIs))
    end
end

#=
The short-circuit "x == y" case in Base.isapprox is bad for us
because it could unnecessarily lead to a boolean-predicate
depends on output error where StochasticAD cannot prove correctness.
We patch up the rule by removing the short-circuit, allowing some common
cases to work.

In the future, we will ideally handle the overloading rule in a more general
way. (E.g. by catching the chain rule for isapprox and recursively calling isapprox
on the values.)
=#
function Base.isapprox(st1::StochasticTriple, st2::StochasticTriple;
                       atol::Real = 0, rtol::Real = Base.rtoldefault(st1, st2, atol),
                       nans::Bool = false, norm::Function = abs)
    (isfinite(st1) && isfinite(st2) &&
     norm(st1 - st2) <= max(atol, rtol * max(norm(st1), norm(st2)))) ||
        (nans && isnan(st1) && isnan(st2))
end
function Base.isapprox(st1::StochasticTriple, x::Real;
                       atol::Real = 0, rtol::Real = Base.rtoldefault(st1, x, atol),
                       nans::Bool = false, norm::Function = abs)
    (isfinite(st1) && isfinite(x) &&
     norm(st1 - x) <= max(atol, rtol * max(norm(st1), norm(x)))) ||
        (nans && isnan(st1) && isnan(x))
end
function Base.isapprox(x::Real, st::StochasticTriple; kwargs...)
    return Base.isapprox(st, x; kwargs...)
end

"""
    Base.getindex(C::AbstractArray, st::StochasticTriple{T})

A simple prototype rule for array indexing. Assumes that underlying type of `st` can index into collection C.
"""
# TODO: support multiple indices, cartesian indices, non abstract array indexables, other use cases...
function Base.getindex(C::AbstractArray, st::StochasticTriple{T}) where {T}
    val = C[st.value]
    function do_map(Δ)
        return value(C[st.value + Δ]) - value(val)
    end
    Δs = map(do_map, st.Δs)
    if val isa StochasticTriple
        Δs = combine((Δs, val.Δs))
    end
    return StochasticTriple{T}(value(val), delta(val), Δs)
end
