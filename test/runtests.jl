using SafeTestsets
using Test
import Random

Random.seed!(1234)

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP in ("All", "Core")
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
