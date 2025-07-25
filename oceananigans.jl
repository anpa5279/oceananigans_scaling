using Pkg
using MPI

MPI.Init()

using CUDA
using Random
using Statistics
using Printf
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth

setup_start = time()
const Nx = 384        # number of points in each of x direction
const Ny = 384        # number of points in each of y direction
const Nz = 384        # number of points in the vertical direction
const Lx = 320    # (m) domain horizontal extents
const Ly = 320    # (m) domain horizontal extents
const Lz = 96    # (m) domain depth 
const N² = 5.3e-9    # s⁻², initial and bottom buoyancy gradient
const initial_mixed_layer_depth = 30.0 # m 
const Q = 0.0     # W m⁻², surface heat flux. cooling is positive
const cᴾ = 4200.0    # J kg⁻¹ K⁻¹, specific heat capacity of seawater
const ρₒ = 1026.0    # kg m⁻³, average density at the surface of the world ocean
const dTdz = 0.01  # K m⁻¹, temperature gradient
const T0 = 25.0    # C, temperature at the surface  
const S0 = 35.0    # ppt, salinity 
const β = 2.0e-4     # 1/K, thermal expansion coefficient
const u₁₀ = 5.75   # (m s⁻¹) wind speed at 10 meters above the ocean
const La_t = 0.3  # Langmuir turbulence number

# Determine architecture based on number of MPI ranks
Nranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Nranks > 1 ? Distributed(GPU()) : GPU()

# Determine rank safely depending on architecture
rank = arch isa Distributed ? arch.local_rank : 0
Nranks = arch isa Distributed ? MPI.Comm_size(arch.communicator) : 1

println("Hello from process $rank out of $Nranks")

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

include("stokes.jl")

#stokes drift
dusdz = Field{Center, Center, Center}(grid)
z_indices = axes(dusdz, 3)
z1d = getindex.(Ref(grid.z.cᵃᵃᶜ), z_indices)
dusdz_1d = dstokes_dz.(z1d, u₁₀)
set!(dusdz, dusdz_1d)
us = stokes_velocity(z1d[end], u₁₀)
@show dusdz

#BCs
u_f = La_t^2 * us
@show u_f
τx = -(u_f^2) #τx = -3.72e-5
u_f = sqrt(abs(τx))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
#@show buoyancy
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Q / (cᴾ * ρₒ * Lx * Ly)),
                                bottom = GradientBoundaryCondition(dTdz))
coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = Smagorinsky(coefficient=0.1),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))
@show model

# ICs
r_z(z) = randn(Xoshiro()) * exp(z/4)
Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * r_z(z) 
uᵢ(x, y, z) = u_f * 1e-1 * r_z(z) 
vᵢ(x, y, z) = -u_f * 1e-1 * r_z(z) 
set!(model, u=uᵢ, v=vᵢ, T=Tᵢ)

@show "equations defined"
set!(model, T=Tᵢ) #u=uᵢ, w=wᵢ, 
simulation = Simulation(model, Δt=30.0, stop_time = 10minutes) #stop_time = 96hours,
@show simulation
wall_clock = Ref(time_ns())

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl=0.5, max_Δt=30seconds)

output_interval = 5minutes

u, v, w = model.velocities
T = model.tracers.T
W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))

simulation.output_writers[:fields] = JLD2Writer(model, (; u, v, w, T),
                                                    schedule = TimeInterval(output_interval),
                                                    filename = "fields.jld2", #$(rank)
                                                    overwrite_existing = true,
                                                    init = save_IC!)

W = Average(w, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
T = Average(T, dims=(1, 2))
                                                      
simulation.output_writers[:averages] = JLD2Writer(model, (; U, V, W, T),
                                                    schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                    filename = "averages.jld2",
                                                    overwrite_existing = true)
wall_clock = Ref(time_ns())
function progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg


    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
#include("nan-check.jl")
#simulation.callbacks[:nan_checker] = Callback(num_check, IterationInterval(1)) #Callback(NaNChecker(fields_to_output, true), IterationInterval(1))

setup_end = time()
# MPI.Barrier()
start1 = time()
run!(simulation)
end1 = time()
if rank == 0
    @show setup_end - setup_start
    @show start1
    @show end1
    @show end1 - start1
    @show simulation.model.clock.iteration
end

MPI.Finalize()