using SafeTestsets
using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin if GROUP == "All"
    @time @safetestset "Distributions" begin include("triples.jl") end
end end
