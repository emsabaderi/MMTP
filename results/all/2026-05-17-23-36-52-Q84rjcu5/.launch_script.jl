using Serialization
using Pigeons
include(raw"/home/easbr/.julia/dev/MMTP/utils/FS2000_PT_setup.jl")
Pigeons.mpi_active_ref[] = true

pt_arguments = 
    try
        Pigeons.deserialize_immutables!(raw"/home/easbr/.julia/dev/MMTP/results/all/2026-05-17-23-36-52-Q84rjcu5/immutables.jls")
        deserialize(raw"/home/easbr/.julia/dev/MMTP/results/all/2026-05-17-23-36-52-Q84rjcu5/.pt_argument.jls")
    catch e
        println("Hint: probably missing dependencies, use the dependencies argument in MPIProcesses() or ChildProcess()")
        rethrow(e)
    end

pt = PT(pt_arguments, exec_folder = raw"/home/easbr/.julia/dev/MMTP/results/all/2026-05-17-23-36-52-Q84rjcu5")
pigeons(pt)
