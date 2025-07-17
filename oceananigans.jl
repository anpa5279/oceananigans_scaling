using Pkg
using MPI

MPI.Init()

using CUDA
using Random
using Statistics
using Printf
using Oceananigans
using Oceananigans.Models: NaNChecker
using Oceananigans.DistributedComputations
using Oceananigans.Units: minute, minutes, hours, seconds
using Oceananigans.BuoyancyFormulations: g_Earth
#import Oceananigans.BoundaryConditions: #fill_halo_regions!

setup_start = time()
const Nx = 512        # number of points in each of x direction
const Ny = 512        # number of points in each of y direction
const Nz = 512        # number of points in the vertical direction
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

#@show dusdz

u_f = La_t^2 * us
@show u_f
τx = -(u_f^2) #τx = -3.72e-5# -(u_f^2)
u_f = sqrt(abs(τx))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))
#@show u_bcs

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 2e-4), constant_salinity = 35.0)
#@show buoyancy
T_bcs = FieldBoundaryConditions(top = GradientBoundaryCondition(0.0),
                                bottom = GradientBoundaryCondition(dTdz))
coriolis = FPlane(f=1e-4) # s⁻¹

model = NonhydrostaticModel(; grid, buoyancy, coriolis,
                            advection = WENO(),
                            timestepper = :RungeKutta3,
                            tracers = (:T),
                            closure = AnisotropicMinimumDissipation(),
                            stokes_drift = UniformStokesDrift(∂z_uˢ=dusdz),
                            boundary_conditions = (u=u_bcs, T=T_bcs))
@show model

# random seed
#rng = Xoshiro(1234 + rank)

#Ξ(x, y, z) = randn(rng) * exp(z/4)

Tᵢ(x,y,z) = T0 - dTdz * (z + initial_mixed_layer_depth)#Tᵢ(x, y, z) = z > - initial_mixed_layer_depth ? T0 : T0 + dTdz * (z + initial_mixed_layer_depth)+ dTdz * model.grid.Lz * 1e-6 * Ξ(x, y, z)
uᵢ(x, y, z) = u_f * 1e-1 * Ξ(x, y, z)
wᵢ(x, y, z) = u_f * 1e-1 * Ξ(x, y, z)
@show "equations defined"
set!(model, u=uᵢ, w=wᵢ,T=Tᵢ) #u=uᵢ, w=wᵢ, 
# After set! calls:
#fill_halo_regions!(dusdz)
#fill_halo_regions!(model.velocities.u)
#fill_halo_regions!(model.velocities.v)
#fill_halo_regions!(model.velocities.w)
#fill_halo_regions!(model.tracers.T)
simulation = Simulation(model, Δt=30.0, stop_time = 0.5hours) #stop_time = 96hours,
@show simulation
wall_clock = Ref(time_ns())

conjure_time_step_wizard!(simulation, cfl=0.5, max_Δt=30seconds)


output_interval = 10minutes

fields_to_output = merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = JLD2Writer(model, fields_to_output,
                                                      schedule = TimeInterval(output_interval),
                                                      filename = "scaling_test_fields_$(rank).jld2",
                                                      overwrite_existing = true)
wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))
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