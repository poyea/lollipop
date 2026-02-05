/*
 *  Sparse matrix-vector multiply (SpMV) — CSR format.
 *
 *  Compressed Sparse Row (CSR) stores a sparse matrix using three arrays:
 *      row_ptr[i]  — index into col_idx/values where row i starts
 *      col_idx[k]  — column index of the k-th non-zero
 *      values[k]   — value of the k-th non-zero
 *
 *  Row i has non-zeros at positions row_ptr[i] .. row_ptr[i+1]-1.
 *
 *  Each thread computes one element of the output vector y:
 *      y[i] = sum_k  A[i, col_idx[k]] * x[col_idx[k]]
 *           = sum_{k = row_ptr[i]}^{row_ptr[i+1]-1}  values[k] * x[col_idx[k]]
 *
 *  This is the simplest SpMV kernel.  Memory access into x[] is irregular
 *  (indirect / gather pattern) — a key difference from dense kernels where
 *  accesses are contiguous.  Real-world SpMV is often memory-bandwidth
 *  bound due to these scattered reads.
 *
 *  Parameters:
 *      row_ptr — (num_rows + 1) int32 array
 *      col_idx — nnz int32 array  (column indices)
 *      values  — nnz float32 array (non-zero values)
 *      x       — input vector  (num_cols float32)
 *      y       — output vector (num_rows float32)
 *      num_rows — number of rows in the sparse matrix
 *
 *  Launch: block=(256,), grid=((num_rows+255)/256,)
 */
extern "C" __global__
void sparse_matvec(const int* row_ptr, const int* col_idx, const float* values,
                   const float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float dot = 0.0f;
    int start = row_ptr[row];
    int end   = row_ptr[row + 1];

    for (int k = start; k < end; k++) {
        dot += values[k] * x[col_idx[k]];
    }

    y[row] = dot;
}
