using Serialization
using Pigeons
include(raw"/home/emad/.julia/dev/MMTP/utils/FS2000_PT_setup.jl")
Pigeons.mpi_active_ref[] = true

pt_arguments = 
    try
        Pigeons.deserialize_immutables!(raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-18-13-56-40-sOIrHsz3/immutables.jls")
        deserialize(raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-18-13-56-40-sOIrHsz3/.pt_argument.jls")
    catch e
        println("Hint: probably missing dependencies, use the dependencies argument in MPIProcesses() or ChildProcess()")
        rethrow(e)
    end

pt = PT(pt_arguments, exec_folder = raw"/home/emad/.julia/dev/MMTP/results/all/2026-05-18-13-56-40-sOIrHsz3")
pigeons(pt)
