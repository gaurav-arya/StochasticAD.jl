module Utils

export print_group

using Functors
using BenchmarkTools

## Printing

# Type piracy, fine since just in benchmarking. (design of Functors should probably allow for user-customized functors)
@functor BenchmarkTools.BenchmarkGroup

function print_trial(t)
    ptime = BenchmarkTools.prettytime(time(t))
    pallocs = "$(allocs(t)) allocs"
    return "$ptime, $pallocs"
end

function print_group(b)
    fmap(t -> (t isa BenchmarkTools.Trial ? print_trial(t) : t), b)
end

end
