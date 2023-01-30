module GoLCore

using Random
using Distributions
using LinearAlgebra
using StochasticAD
using StaticArrays
using OffsetArrays

function update_state!(all_probs, N, board, board_old)
    for i in (-N):N
        for j in (-N):N
            neighbours = board_old[i + 1, j] + board_old[i - 1, j] + board_old[i, j - 1] +
                         board_old[i, j + 1]
            index = board[i, j] * 5 + neighbours + 1 # trick necessary because we do not have implementation support for stochastic triple not <: Real
            b = rand(Bernoulli(all_probs[index]))
            board[i, j] += (1 - 2 * board[i, j]) * b
        end
    end
end

function play_game_of_life(p, all_probs, N, T, log = false)
    dual_type = promote_type(typeof(rand(Bernoulli(p))),
                             typeof.(rand.(Bernoulli.(all_probs)))...) # TODO: better way of getting the concrete type
    board = OffsetArray(zeros(dual_type, 2 * N + 3, 2 * N + 3), (-(N + 1)):(N + 1),
                        (-(N + 1)):(N + 1)) # pad by 1
    for i in (-N):N
        for j in (-N):N
            board[i, j] = rand(Bernoulli(p))
        end
    end
    board_old = similar(board)
    log && (history = [])
    for time_step in 1:T
        copy!(board_old, board)
        update_state!(all_probs, N, board, board_old)
        log && push!(history, copy(board))
    end
    if !log
        return sum(board)
    else
        return sum(board), board, history
    end
end

function play(p, θ = 0.1, N = 3, T = 3; log = false)
    # N is the board half-length, T are game time steps
    low = θ
    high = 1 - θ
    birth_probs = SA[low, low, low, high, low] # 0, 1, 2, 3, 4 neighbours
    death_probs = SA[high, high, low, low, high] # 0, 1, 2, 3, 4 neighbours
    return play_game_of_life(p, vcat(birth_probs, death_probs), N, T, log)
end

# An implementation of finite differences that uses "common random numbers"
# (the same seed), for more accurate checking, albeit with a finite step size h
# such that there is weight degeneracy as h → 0.
function fd_clever(p, h = 0.001)
    state = copy(Random.default_rng())
    run1 = play(p + h)
    copy!(Random.default_rng(), state)
    run2 = play(p - h)
    (run1 - run2) / (2h)
end

# Provide some default parameters
p = 0.5
nsamples = 200_000

end
