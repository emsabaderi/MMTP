using Serialization
using Pigeons
include(raw"/home/emad/.julia/dev/MMTP/utils/FS2000_PT_setup.jl")
Pigeons.mpi_active_ref[] = true

pt_arguments = 
    try
        Pigeons.deserialize_immutables!(raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-23-12-56-45-eMGOXjDt/immutables.jls")
        deserialize(raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-23-12-56-45-eMGOXjDt/.pt_argument.jls")
    catch e
        println("Hint: probably missing dependencies, use the dependencies argument in MPIProcesses() or ChildProcess()")
        rethrow(e)
    end

pt = PT(pt_arguments, exec_folder = raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-23-12-56-45-eMGOXjDt")
pigeons(pt)
