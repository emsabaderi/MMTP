# %% helper for serialized samples
function cache_or_compute(compute_fn, path::AbstractString; force::Bool=false)
    if !force && isfile(path)
        @info "Loading cached result from $path"
        return deserialize(path)
    end
    @info "Computing fresh result and caching to $path"
    mkpath(dirname(path))
    result = compute_fn()
    serialize(path, result)
    return result
end
