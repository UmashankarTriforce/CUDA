function DGEMM(A, B)    
    rowA, colA = size(A)
    rowB, colB = size(B)

    if(size(A)[2] == size(B)[1])
        C = ccall((:DGEMM, "./libdgemm.so"), Ptr{Cdouble}, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Cint), A, B, rowA, colA, rowB, colB)    
        return unsafe_wrap(Array{Float64}, C, (rowA, colB), own = true)
    end
end
        