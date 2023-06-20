using SafeTestsets
using Test, Pkg
import Random

Random.seed!(1234)

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

@time begin
    if GROUP == "All"
        @time @safetestset "Triples" begin
            include("triples.jl")
        end
        @time @safetestset "Game of life" begin
            include("game_of_life.jl")
        end
        @time @safetestset "Random walk" begin
            include("random_walk.jl")
        end
        @time @safetestset "Resampling" begin
            include("resampling.jl")
        end
    end
end
