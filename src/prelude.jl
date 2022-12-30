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

const UNARY_TYPEFUNCS_NOWRAP = [:(rtoldefault)]
const UNARY_TYPEFUNCS_WRAP = [
    :(typemin),
    :(typemax),
    :(floatmin),
    :(floatmax),
    :(zero),
    :(one),
]
const RNG_TYPEFUNCS_WRAP = [:(rand), :(randn), :(randexp)]
