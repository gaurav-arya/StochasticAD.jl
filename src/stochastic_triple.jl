""" 
    StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}}

Stores the primal value of the computation, alongside a "dual" component
representing an infinitesimal change, and a "triple" component that tracks
discrete change(s) with infinitesimal probability. 

Pretty printed as "value + δε + ({pretty print of Δs})".

## Constructor

- `value`: the primal value.
- `δ`: the value of the almost-sure derivative, i.e. the rate of "infinitesimal" change.
- `Δs`: alternate values with associated weights, i.e. Finite perturbations with Infinitesimal probability,
        represented by a backend `FIs <: AbstractFIs`.
"""
struct StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}} <: Real
    value::V
    δ::V # infinitesimal change
    Δs::FIs # finite changes with infinitesimal probabilities # (Δ = 3, p = 1*h)
    function StochasticTriple{T, V, FIs}(value::V, δ::V,
                                         Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
        new{T, V, FIs}(value, δ, Δs)
    end
end

"""
    value(st::StochasticTriple)

Return the primal value of `st`.
"""
value(x::Real) = x
value(st::StochasticTriple) = st.value

"""
    delta(st::StochasticTriple)

Return the almost-sure derivative of `st`, i.e. the rate of infinitesimal change.
"""
delta(x::Real) = zero(x)
delta(st::StochasticTriple) = st.δ

"""
    perturbations(st::StochasticTriple)

Return the finite perturbation(s) of `st`, in a format dependent on the [backend](devdocs.md) used for storing perturbations.
"""
perturbations(x::Real) = ()
perturbations(st::StochasticTriple) = perturbations(st.Δ)

"""
    derivative_contribution(st::StochasticTriple)

Return the derivative estimate given by combining the dual and triple components of `st`.
"""
derivative_contribution(x::Real) = zero(x)
derivative_contribution(st::StochasticTriple) = st.δ + derivative_contribution(st.Δs)

### Extra constructors of stochastic triples

function StochasticTriple{T}(value::V, δ::V, Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
    StochasticTriple{T, V, FIs}(value, δ, Δs)
end

function StochasticTriple{T}(value::A, δ::B,
                             Δs::FIs) where {T, A, B, C, FIs <: AbstractFIs{C}}
    V = promote_type(A, B, C)
    StochasticTriple{T}(convert(V, value), convert(V, δ), similar_type(FIs, V)(Δs))
end

### Conversion rules

# TODO: is this the right thing to do? Maybe, different from the promote case because there V was guaranteed to be an ancestor. 
# Also, bad to do when already same type?
function Base.convert(::Type{StochasticTriple{T, V, FIs}},
                      x::StochasticTriple{T}) where {T, V, FIs}
    StochasticTriple{T, V, FIs}(convert(V, x.value), convert(V, x.δ), FIs(x.Δs))
end

# TODO: ForwardDiff's promotion rules are a little more complicated, see https://github.com/JuliaDiff/ForwardDiff.jl/issues/322
# May need to look into why and possibly use them here too.
function Base.promote_rule(::Type{StochasticTriple{T, V1, FIs}},
                           ::Type{StochasticTriple{T, V2, FIs2}}) where {T, V1, FIs, V2,
                                                                         FIs2}
    V = promote_type(V1, V2)
    StochasticTriple{T, V, similar_type(FIs, V)}
end

function Base.promote_rule(::Type{StochasticTriple{T, V1, FIs}},
                           ::Type{V2}) where {T, V1, FIs, V2 <: Real}
    V = promote_type(V1, V2)
    StochasticTriple{T, V, similar_type(FIs, V)}
end

Base.hash(st::StochasticAD.StochasticTriple) = hash(StochasticAD.value(st)) # TODO: port

for op in UNARY_TYPEFUNCS
    @eval function $op(::Type{StochasticAD.StochasticTriple{T, V, FIs}}) where {T, V, FIs}
        return StochasticAD.StochasticTriple{T, V, FIs}($op(V), zero(V), empty(FIs))
    end
end

Base.zero(st::StochasticTriple) = zero(typeof(st))
Base.one(st::StochasticTriple) = one(typeof(st))

function Base.convert(::Type{StochasticTriple{T, V, FIs}}, x::Real) where {T, V, FIs}
    StochasticTriple{T, V, FIs}(convert(V, x), zero(V), empty(FIs))
end

### Creating the first stochastic triple in a computation

function StochasticTriple{T}(value::V, δ::V, backend::Type{<:AbstractFIs}) where {T, V}
    StochasticTriple{T}(value, δ, similar_type(backend, V)())
end

function StochasticTriple{T}(value::V, backend::Type{<:AbstractFIs}) where {T, V}
    StochasticTriple{T}(value, zero(V), backend)
end

function StochasticTriple{T}(value::A, δ::B, backend::Type{<:AbstractFIs}) where {T, A, B}
    V = promote_type(A, B)
    StochasticTriple{T}(convert(V, value), convert(V, δ), backend)
end

### Showing a stochastic triple

function Base.summary(::StochasticTriple{T, V}) where {T, V}
    return "StochasticTriple of $V"
end

function Base.show(io::IO, ::MIME"text/plain", st::StochasticTriple)
    println(io, "$(summary(st)):")
    show(io, st)
end

function Base.show(io::IO, st::StochasticTriple)
    print(io, "$(st.value) + $(st.δ)ε")
    if (!isempty(st.Δs))
        print(io, " + ($(repr(st.Δs)))")
    end
end

### Higher level functions

struct Tag{F, V}
end

"""
    stochastic_triple(X, p; backend=StochasticAD.PrunedFIs)
    stochastic_triple(p; backend=StochasticAD.PrunedFIs)

If `p <: Real`, return the result of propagating the stochastic triple `p + ε` through the random function `X(p)`.
If `p <: AbstractVector`, return a vector of stochastic triples of the same shape as `p`, containing the stochastic
triples that result from perturbing the corresponding array elements of `p` one-by-one.
When `X` is not provided, the identity function is used. The `backend` keyword argument describes the algorithm 
used by the third component of the stochastic triple, see [technical details](devdocs.md) for more details.
"""
function stochastic_triple(f, p::V; backend = PrunedFIs) where {V <: Real}
    st = StochasticTriple{Tag{typeof(f), V}}(p, one(p), backend)
    return f(st)
end

function stochastic_triple(f, p::AbstractVector{V}; backend = PrunedFIs) where {V}
    function map_func(perturbed_index)
        sts = map(eachindex(p), p) do i, p_i
            if i == perturbed_index
                return StochasticTriple{Tag{typeof(f), V}}(p_i, one(p_i), backend)
            else
                return StochasticTriple{Tag{typeof(f), V}}(p_i, zero(p_i), backend)
            end
        end
        return f(sts)
    end
    return map(map_func, eachindex(p))
end

stochastic_triple(p; kwargs...) = stochastic_triple(x -> x, p; kwargs...)

@doc raw"""
    derivative_estimate(X, p; backend=StochasticAD.PrunedFIs)

Compute an unbiased estimate of ``\frac{\mathrm{d}\mathbb{E}[X(p)]}{\mathrm{d}p}``, the derivative of the expectation of the real-valued random function `X(p)` 
with respect to its input `p`, where `p <: Real` or `p <: AbstractVector`.
The `backend` keyword argument describes the algorithm used by the third component of the stochastic triple, see [technical details](devdocs.md) for more details.
"""
function derivative_estimate(f, p::Real; kwargs...)
    derivative_contribution(stochastic_triple(f, p; kwargs...))
end

function derivative_estimate(f, p::AbstractVector; kwargs...)
    derivative_contribution.(stochastic_triple(f, p; kwargs...))
end
