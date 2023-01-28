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

function StochasticTriple{T}(value::V, Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
    StochasticTriple{T}(value, zero(value), Δs)
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

For any `p` that is supported by [`Functors.jl`]() (e.g. scalars, arrays), 
return an output of similar structure to p, where a particular value contains
the stochastic-triple output of `X` when perturbing the corresponding value in `p`
(i.e. replacing the original value `x` with the triple `x + ε`).
When `X` is not provided, the identity function is used. 

The `backend` keyword argument describes the algorithm used by the third component
of the stochastic triple, see [technical details](devdocs.md) for more details.

# Example
```jldoctest
julia> using Distributions, Random, StochasticAD; Random.seed!(4321);

julia> stochastic_triple(rand ∘ Bernoulli, 0.5)
StochasticTriple of Int64:
0 + 0ε + (1 with probability 2.0ε, tag 1)
```
"""
function stochastic_triple(f, p::V; backend = PrunedFIs) where {V}
    counter = begin
        c = 0
        (_) -> begin
            c += 1
            return c
        end
    end
    indices = structural_map(counter, p)
    function map_func(perturbed_index)
        sts = structural_map(indices, p) do i, p_i
            if i == perturbed_index
                return StochasticTriple{Tag{typeof(f), V}}(p_i, one(p_i), backend)
            else
                return StochasticTriple{Tag{typeof(f), V}}(p_i, zero(p_i), backend)
            end
        end
        out = f(sts)
        return out
    end
    return structural_map(map_func, indices)
end

stochastic_triple(p; kwargs...) = stochastic_triple(x -> x, p; kwargs...)

@doc raw"""
    derivative_estimate(X, p; backend=StochasticAD.PrunedFIs)

Compute an unbiased estimate of ``\frac{\mathrm{d}\mathbb{E}[X(p)]}{\mathrm{d}p}``, 
the derivative of the expectation of the random function `X(p)` with respect to its input `p`. 

Both `p` and `X(p)` can be any object supported by [`Functors.jl`](https://fluxml.ai/Functors.jl/stable/),
e.g. scalars or abstract arrays. 
The output of `derivative_estimate` has the same outer structure as `p`, but with each
scalar in `p` replaced by a derivative estimate of `X(p)` with respect to that entry.

For example, if `X(p) <: AbstractMatrix` and `p <: Real`, then the output would be a matrix.
The `backend` keyword argument describes the algorithm used by the third component
of the stochastic triple, see [technical details](devdocs.md) for more details.

!!! note 
    Since `derivative_estimate` performs forward-mode AD, the required computation time scales
    linearly with the number of parameters in `p` (but is unaffected by the number of parameters in `X(p)`).
# Example
```jldoctest
julia> using Distributions, Random, StochasticAD; Random.seed!(4321);

julia> derivative_estimate(rand ∘ Bernoulli, 0.5) # A random quantity that averages to 1.
2.0

julia> derivative_estimate(x -> [rand(Bernoulli(x * i/100)) for i in 1:100], 0.5)
100-element Vector{Float64}:
 0.010050251256281407
 0.020202020202020204
 0.03045685279187817
 0.04081632653061225
 ⋮
 0.0
 0.0
 2.0
```
"""
function derivative_estimate(f, p; kwargs...)
    StochasticAD.structural_map(derivative_contribution, stochastic_triple(f, p; kwargs...))
end
