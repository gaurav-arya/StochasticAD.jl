const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

const UNARY_PREDICATES = [isinf, isnan, isfinite, iseven, isodd, isreal, isinteger]

const BINARY_PREDICATES = [
    isequal,
    isless,
    Base.:<,
    Base.:>,
    Base.:(==),
    Base.:(!=),
    Base.:(<=),
    Base.:(>=),
]

const UNARY_TYPEFUNCS_NOWRAP = [:(Base.hash), :(Base.rtoldefault)]
const UNARY_TYPEFUNCS_WRAP = [
    :(Base.typemin),
    :(Base.typemax),
    :(Base.floatmin),
    :(Base.floatmax),
    :(Base.zero),
    :(Base.one),
]
const RNG_TYPEFUNCS_WRAP = [:(Random.rand), :(Random.randn), :(Random.randexp)]
