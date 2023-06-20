include("core.jl")
include("model.jl")
using Plots, LaTeXStrings

# visualization of stochastic process (observations and latent states), particle filter, and Kalman filter

### run and visualize filters
Xs, W = particle_filter(θtrue; store_path = true)
fig = plot(getindex.(xs, 1), getindex.(xs, 2), legend = false, xlabel = L"x_1",
    ylabel = L"x_2") # x1 and x2 are bad names..conflicting notation
scatter!(fig, getindex.(ys, 1), getindex.(ys, 2))
for i in 1:min(m, 100) # note that Xs has obs noise.
    local xs = [Xs[t][i] for t in 1:T]
    scatter!(fig, getindex.(xs, 1), getindex.(xs, 2), marker_z = 1:T, color = :cool,
        alpha = 0.1) # color to indicate time step
end

xs_Kalman, ll_Kalman = kalman_filter(θtrue)
plot!(getindex.(mean.(xs_Kalman), 1), getindex.(mean.(xs_Kalman), 2), legend = false,
    color = "red")
display(fig)
savefig(fig, "filter.pdf")
