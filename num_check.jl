function num_check(sim)
    fields_to_check = merge(sim.model.velocities, sim.model.tracers, sim.model.pressures, (; νₑ=sim.model.diffusivity_fields.νₑ, κₑ=sim.model.diffusivity_fields.κₑ.T, ∂z_u =sim.model.stokes_drift.∂z_uˢ))
    n_fields = keys(fields_to_check)
    #@show n_fields
    for (i, name) in enumerate(n_fields)
        #@show name
        field = fields_to_check[name]
        #@show field
        x = field.data
        if any(isnan.(x))
            index = collect.(Tuple.(findall(isnan.(x))))
            msg = @sprintf("iteration: %d, time: %s, NaN in field %s at: %d, %d, %d \n",
                            iteration(sim), 
                            prettytime(time(sim)), 
                            "$(n_fields[i])",
                            index[1][1], 
                            index[1][2], 
                            index[1][3])
            @info msg
            error()

        elseif any(isinf.(x))
            index = collect.(Tuple.(findall(isinf.(x))))
            msg = @sprintf("iteration: %d, time: %s, inf in field %s at: %d, %d, %d\n",
                            iteration(sim), 
                            prettytime(time(sim)), 
                            "$(n_fields[i])",
                            index[1][1], 
                            index[1][2], 
                            index[1][3])
            @info msg
            error()
        end
    end
    
    return nothing
end