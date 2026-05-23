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

function pt_cache_or_compute(compute_fn::Function, folder::String; force::Bool=false)
    if !force && isfile(joinpath(folder, "immutables.jls"))
        @info "Loading cached PT from $folder"
        return Pigeons.PT(folder)
    end

    @info "Computing fresh PT result; will cache to $folder"
    isdir(folder) && rm(folder; recursive=true)

    result = compute_fn()           # `Result{PT}` (or `PT` if no ChildProcess)
    cp(result.exec_folder, folder)  # snapshot to stable location

    return Pigeons.PT(folder)
end
