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

const UNARY_TYPEFUNCS = [:(Base.typemin), :(Base.typemax), :(Base.floatmin), :(Base.floatmax), :(Base.zero), :(Base.one)]