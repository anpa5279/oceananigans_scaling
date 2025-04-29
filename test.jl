using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
size = MPI.Comm_size(MPI.COMM_WORLD)

ranks = zeros(Int, size)
ranks[rank+1] = rank + 1

MPI.Allreduce!(ranks, +, MPI.COMM_WORLD)

@info rank ranks

MPI.Finalize()
