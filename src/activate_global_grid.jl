export activate_global_grid

"""
    old_gg = activate_global_grid(new_grid)

Replaces the current active grid parameters with the ones provided by the new_grid. The previously active grid is returned. Only one grid can be active at the same time.


# Argument
- `new_gg::GlobalGrid`: the global grid configuration to be set active. It must be a GlobalGrid returned by a call to `create_global_grid`.

# Return value
- `old_gg::GlobalGrid`: the previously active global grid configuration that has been substituted. If this is the first activation, a Nothing will be returned. 

# Usage example
    Given two local domains of different size and/or ghost cell properties: bigger_array_A, bigger_array_B and two grids that reflect those differences respectively: A,B

    activate_global_grid(A)
    update_halo!(bigger_array_A)
    activate_global_grid(B)
    update_halo!(smaller_array_B)

See also: [`init_global_grid`](@ref), [`create_global_grid`](@ref)
"""
function activate_global_grid(new_gg :: GlobalGrid) :: Union{GlobalGrid, Nothing}
    check_initialized()
    old_gg = get_global_grid()
    set_global_grid(new_gg)
    return grid_is_initialized() ? old_gg : Nothing
end