function num_check(sim)
    fields_to_check = merge(sim.model.velocities, sim.model.tracers, sim.model.pressures, (; νₑ=sim.model.diffusivity_fields.νₑ, κₑ=sim.model.diffusivity_fields.κₑ.T, ∂z_u =sim.model.stokes_drift.∂z_uˢ))
    n_fields = keys(fields_to_check)
    for (n, name) in enumerate(n_fields)
        field = fields_to_check[name]
        x = Array(parent(field))
        Nmax = field.grid.Nz + field.grid.Hz
        if any(isnan, parent(field))
            index = collect.(Tuple.(findall(isnan, x)))
            for i in 1:length(index)
                if 1 in index[i] || Nmax in index[i] 
                    println("outer most halo, not showing index")
                else
                    xp = x[index[i][1]+1, index[i][2], index[i][3]]
                    xm = x[index[i][1]-1, index[i][2], index[i][3]]
                    yp = x[index[i][1], index[i][2]+1, index[i][3]]
                    ym = x[index[i][1], index[i][2]-1, index[i][3]]
                    zp = x[index[i][1], index[i][2], index[i][3]+1]
                    zm = x[index[i][1], index[i][2], index[i][3]-1]
                    msg = @sprintf("iteration: %d, time: %s, NaN in field %s at: %d, %d, %d \n
                                    surrounding values: 
                                    +Δx: %6.3e,\n
                                    -Δx: %6.3e, \n
                                    +Δy: %6.3e,\n
                                    -Δy: %6.3e,\n
                                    +Δz: %6.3e, \n
                                    -Δz: %6.3e, \n",
                                    iteration(sim), 
                                    prettytime(time(sim)), 
                                    "$(n_fields[n])",
                                    index[i][1], 
                                    index[i][2], 
                                    index[i][3], 
                                    xp, xm, yp, ym, zp, zm)                              
                    @info msg 
                    error()
                end
            end
        elseif any(isinf, parent(field))
            index = collect.(Tuple.(findall(isinf, x)))
            for i in 1:length(index)
                if 1 in index[i] || Nmax in index[i] 
                    println("outer most halo, not showing index")
                else
                    msg = @sprintf("iteration: %d, time: %s, Inf in field %s at: %d, %d, %d \n
                                    surrounding values: 
                                    +Δx: %6.3e,\n
                                    -Δx: %6.3e, \n
                                    +Δy: %6.3e,\n
                                    -Δy: %6.3e,\n
                                    +Δz: %6.3e, \n
                                    -Δz: %6.3e, \n",
                                    iteration(sim), 
                                    prettytime(time(sim)), 
                                    "$(n_fields[n])",
                                    index[i][1], 
                                    index[i][2], 
                                    index[i][3], 
                                    x[index[i][1]+1, index[i][2], index[i][3]],
                                    x[index[i][1]-1, index[i][2], index[i][3]],
                                    x[index[i][1], index[i][2]+1, index[i][3]],
                                    x[index[i][1], index[i][2]-1, index[i][3]],
                                    x[index[i][1], index[i][2], index[i][3]+1],
                                    x[index[i][1], index[i][2], index[i][3]-1])
     
                    @info msg
                    error()
                end
            end
        end
    end
    
    return nothing
end
