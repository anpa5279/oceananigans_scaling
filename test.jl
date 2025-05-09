using Pkg
using Statistics
using Printf
using Oceananigans
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth

mutable struct Params
    Nx::Int         # number of points in each of x direction
    Ny::Int         # number of points in each of y direction
    Nz::Int         # number of points in the vertical direction
    Lx::Float64     # (m) domain horizontal extents
    Ly::Float64     # (m) domain horizontal extents
    Lz::Float64     # (m) domain depth 
    N²::Float64     # s⁻², initial and bottom buoyancy gradient
    initial_mixed_layer_depth::Float64 # m 
    Q::Float64      # W m⁻², surface heat flux. cooling is positive
    cᴾ::Float64     # J kg⁻¹ K⁻¹, specific heat capacity of seawater
    ρₒ::Float64     # kg m⁻³, average density at the surface of the world ocean
    dTdz::Float64   # K m⁻¹, temperature gradient
    T0::Float64     # C, temperature at the surface   
    β::Float64      # 1/K, thermal expansion coefficient
    u₁₀::Float64    # (m s⁻¹) wind speed at 10 meters above the ocean
    La_t::Float64   # Langmuir turbulence number
end

#defaults, these can be changed directly below 128, 128, 160, 320.0, 320.0, 96.0
p = Params(8, 8, 8, 320.0, 320.0, 96.0, 5.3e-9, 33.0, 0.0, 4200.0, 1000.0, 0.01, 17.0, 2.0e-4, 5.75, 0.3)

grid = RectilinearGrid(; size=(p.Nx, p.Ny, p.Nz), extent=(p.Lx, p.Ly, p.Lz))

include("stokes.jl")
#stokes drift
z_d = collect(-p.Lz + grid.z.Δᵃᵃᶜ/2 : grid.z.Δᵃᵃᶜ : -grid.z.Δᵃᵃᶜ/2)
dudz = dstokes_dz(z_d, p.u₁₀)
new_dUSDdz = Field{Nothing, Nothing, Center}(grid)
set!(new_dUSDdz, reshape(dudz, 1, 1, :))

u_f = p.La_t^2 * (stokes_velocity(-grid.z.Δᵃᵃᶜ/2, p.u₁₀)[1])
τx = -(u_f^2)
u_f = sqrt(abs(τx))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
@show u_bcs

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
@show buoyancy
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(p.dTdz))
coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=new_dUSDdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs)) 
@show model

# random seed
Ξ(z) = randn() * exp(z / 4)

Tᵢ(x, y, z) = z > - p.initial_mixed_layer_depth ? p.T0 : p.T0 + p.dTdz * (z + p.initial_mixed_layer_depth)+ p.dTdz * model.grid.Lz * 1e-6 * Ξ(z)
uᵢ(x, y, z) = u_f * 1e-1 * Ξ(z)
wᵢ(x, y, z) = u_f * 1e-1 * Ξ(z)

set!(model, u=uᵢ, w=wᵢ, T=Tᵢ)

simulation = Simulation(model, Δt=30.0, stop_time = 4hours) #stop_time = 96hours,
@show simulation
wall_clock = Ref(time_ns())

conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30seconds)


output_interval = 10minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2OutputWriter(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "scaling_test_fields.jld2",
                                                      overwrite_existing = true)
wall_clock = Ref(time_ns())

function progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

function num_check(sim)
    fields_to_check = merge(sim.model.velocities, sim.model.tracers, sim.model.pressures, (; νₑ=sim.model.diffusivity_fields.νₑ, κₑ=sim.model.diffusivity_fields.κₑ.T, ∂z_u =sim.model.stokes_drift.∂z_uˢ))
    n_fields = keys(fields_to_check)
    #@show n_fields
    for (i, name) in enumerate(n_fields)
        #@show name
        field = fields_to_check[name]
        #@show field
        x = field.data
        if isfinite(sum(x)) == false 
            println("error")
            if isnan(sum(x))
                index = collect.(Tuple.(findall(isnan.(x))))
                msg = @sprintf("iteration: %d, time: %s, NaN in field %s at: %d, %d, %d \n",
                                iteration(sim), 
                                prettytime(time(sim)), 
                                "$(n_fields[i])",
                                index[1][1], 
                                index[1][2], 
                                index[1][3])

            elseif isinf(sum(x))
                index = collect.(Tuple.(findall(isinf.(x))))
                msg = @sprintf("iteration: %d, time: %s, inf in field %s at: %d, %d, %d\n",
                                iteration(sim), 
                                prettytime(time(sim)), 
                                "$(n_fields[i])",
                                index[1][1], 
                                index[1][2], 
                                index[1][3])

            else
                msg = @sprintf("iteration: %d, time: %s, unknown in field %s\n",
                                iteration(sim), 
                                prettytime(sim),
                                "$(n_fields[i])")
            end
            @info msg
            error()
        end
    end
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
simulation.callbacks[:num_check] = Callback(num_check, IterationInterval(1))

run!(simulation)