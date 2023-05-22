using LinearAlgebra
using Printf
using Random

using JuMP

using Plots

struct UFLData
    m::Int
    n::Int

    Xc::Matrix{Float64}  # Customer coordinates
    Xf::Matrix{Float64}  # Facility coordinates

    f::Vector{Float64}   # fixed cost (per facility)
    D::Matrix{Float64}   # distance matrix
end

function generate_ufl_instance(m, n; seed::Int=1804, grid_param::Int=1024)   
    # To ensure reproducibility, always use a seed!
    rng = MersenneTwister(seed)
    
    Xc = rand(rng, 0:grid_param, n, 2) ./ grid_param  # Customer coordinates
    Xf = rand(rng, -1:21, m, 2) ./ 20                 # Facility coordinates

    # Sample fixed building costs
    f = rand(10:20, m)

    # Compute distance matrix
    D = zeros(Float64, m, n)
    for j in 1:n, i in 1:m
        D[i, j] = norm(Xf[i, :] - Xc[j, :], 2)
    end
    # Increase the distances to make the problem a little harder
    D .*= 10

    return UFLData(m, n, Xc, Xf, f, D)
end

"""
    plot_ufl(data::UFLData)

Visualize a UFL instance data. Returns a `Plot` object.
"""
function plot_ufl_instance(ufl::UFLData)
    # Since we have coordinates, we can visualize what the instance looks like
    plt = plot(
        legend=:topleft,
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.33),
    )

    scatter!(plt, ufl.Xf[:, 1], ufl.Xf[:, 2], label="Locations", marker=:rect, ms=8, opacity=0.5)
    scatter!(plt, ufl.Xc[:, 1], ufl.Xc[:, 2], label="Customers", marker=:circ, ms=5)
    return plt
end

function plot_ufl_solution(ufl::UFLData, y_, x_)
    plt = plot(
        legend=:topleft,
        ylim=(-0.1, 1.33),
    )

    # Show which customer is served by which facility
    for i in 1:ufl.m
        for j in 1:ufl.n
            if x_[i, j]
                # Plot a line between yi and xj
                plot!(plt, [ufl.Xc[j, 1], ufl.Xf[i, 1]], [ufl.Xc[j, 2], ufl.Xf[i, 2]], label=nothing, lw=0.5, color=:black)
            end
        end
    end

    # Customers
    scatter!(plt, ufl.Xc[:, 1], ufl.Xc[:, 2], label="Customers", marker=:circ, ms=4, color=:red)

    # Facilities (open and closed)
    scatter!(plt, [ufl.Xf[.! y_, 1]], [ufl.Xf[.!y_, 2]],
        color=:lightblue,
        label="Facilities (closed)",
        marker=:rect,
        ms=6
    )
    scatter!(plt, [ufl.Xf[y_, 1]], [ufl.Xf[y_, 2]],
        color=:darkblue,
        label="Facilities (open)",
        marker=:rect,
        ms=6
    )

    return plt
end

function build_ufl_mip(data::UFLData)
    model = Model()  # You can defined a JuMP model without a solver...

    m, n = data.m, data.n

    # Decision variables
    @variable(model, y[1:m], Bin)
    @variable(model, x[1:m, 1:n], Bin)
    
    # Constraints
    @constraint(model, customer_served[j in 1:n], sum(x[:, j]) == 1)
    @constraint(model, facility_open[i in 1:m, j in 1:n], x[i, j] <= y[i])
    
    # Objective function
    @objective(model, Min, dot(data.f, y) + dot(data.D, x))

    return model
end

function build_ufl_master(ufl::UFLData; optimizer)
    mp = Model(optimizer)

    m = ufl.m
    n = ufl.n

    @variable(mp, y[1:m], Bin)
    @variable(mp, θ[1:n] >= 0)

    @constraint(mp, sum(y) >= 1)

    @objective(mp, Min, dot(ufl.f, y) + sum(θ))

    return mp
end

"""
    solve_benders_sp(d::AbstractVector, y; kwargs...)

Solve the Benders sub-problem for customer ``j``.

# Arguments
* `d::AbstractVector`: the vector of distances between the customer and all locations.
* `y::AbstractVector`: the current master solution. May be integer or fractional.

# Returns
* `z::Float`: the optimal objective value of the sub-problem
* `λ::Vector{Float64}`: the sub-problem dual solution, used to form the Benders cut
* `x::Vector{Float64}`: the sub-problem primal solution, for information
"""
function solve_benders_sp(d::AbstractVector, y; kwargs...)
    m = length(d)
    
    sp = direct_model(Gurobi.Optimizer(GRBENV))
    
    @variable(sp, x[1:m] >= 0)  # we don't need the upper bound, because it's implied by sum(x) == 1
    @variable(sp, q[1:m])
    
    @constraint(sp, sum(x) == 1)
    @constraint(sp, benders, q .== y)  # Kevin's trick to make your life easier
    @constraint(sp, x .<= q)
    
    @objective(sp, Min, dot(d, x))
    
    set_silent(sp)
    for (k, v) in kwargs
        set_optimizer_attribute(sp, string(k), v)
    end
    
    optimize!(sp)
    z = objective_value(sp)
    λ = dual.(benders)
    x_ = value.(x)
    
    return z, λ, x_
end

function solve_benders_sp_fast(d::AbstractVector, y; kwargs...)
    m = length(d)
    
    # Find the closest facilities...
    p = sortperm(d)
    # ... then greedily allocate until we hit 1.0
    w  = 1.0  # we record w == 1 - sum(x)
    x_ = zeros(Float64, m)
    d_ = Inf  # Distance to the furthest facility that serves this customer
    z_ = 0.0  # Objective value
    for i in p
        x_[i] += min(y[i], w)
        z_ += x_[i] * d[i]
        w -= x_[i]
        d_ = y[i] > 0 ? d[i] : d_
        w <= 0 && break
    end
    
    # Now we build the dual solution
    λ = zeros(m)
    for i in 1:m
        λ[i] = (d[i] < d_) * (d[i] - d_)
    end
    
    return z_, λ, x_
end
