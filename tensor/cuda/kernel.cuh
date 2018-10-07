#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h> 
#include <curand_kernel.h>
#include <curand.h>

#include "MyLib.h"

void Fadd_impl(const dtype* x, const dtype* y,  dtype* r, int size);

void Fadd_impl(dtype** x, dtype** y,  dtype** r, int dim0, int size);

//void Fadd_inplace_impl(dtype* x, dtype** y, int x_size, int size);

void Fadd_inplace_impl(dtype* x, const dtype *y, int size);

void Fadd_inplace_impl(dtype** x, dtype** y, int dim0, int size);

void Dadd_inplace_impl(dtype* in_loss, const dtype* loss, int dim, int n);

void Dadd_inplace_impl(dtype** in_loss, const dtype* loss, int dim, int n);

void Fadd_impl(dtype*** x, dtype** y, int count, int n, int dim0);

void Dadd_impl(dtype*** in_loss, dtype** loss, int count, int n, int dim0);

void Fsubtract_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fsubtract_inplace_impl(dtype* x, const dtype* y, int size);

void Fmultiply_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fmultiply_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fmultiply_inplace_impl(dtype* x, const dtype* y, int size);

void Fmultiply_inplace_impl(dtype** x, dtype** y, int dim0, int size);

void Fdivide_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fmultiply_scalar_impl(const dtype* x, const dtype* scalar, dtype* r, int size);

void Fmultiply_scalar_impl(const dtype* x, const dtype y, dtype* r, int size);

void Fmultiply_scalar_inplace_impl(dtype* x, const dtype y, int size);

void Fadd_scalar_impl(const dtype* x, const dtype y, dtype* r, int size);

void Fadd_scalar_inplace_impl(dtype* x, const dtype y, int size);

void Ddot_impl(dtype **val, dtype **loss, dtype **in_loss, int dim0, int n);

void Fsquare_impl(const dtype* x, dtype* r, int size);

void Fsqrt_impl(const dtype* x, dtype* r, int size);

void Ftanh_impl(const dtype* x, dtype* r, int size);

void Dtanh_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Ftanh_impl(dtype** x, dtype** r, int dim0, int size);

void Dtanh_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fsigmoid_impl(dtype** x, dtype** r, int dim0, int size);

void Dsigmoid_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fsigmoid_impl(const dtype* x, dtype* r, int size);

void Dsigmoid_impl(const dtype* x, const dtype* y, dtype* r, int size);

void concat_impl(dtype **src, dtype* dst, int src_dim, int dst_dim);

void unconcat_impl(const dtype *src, dtype** dst, int src_dim, int dst_dim);

void Fconcat_impl(dtype ***src, int count, int n, int *offset_ptr, int *dims, int max_dim, dtype **dst);

void concat_impl(dtype ***src, int count, int n, int *offset_ptr, int *dims, int y_dim, int max_dim, dtype *dst);

void Dconcat_impl(dtype **loss, int count, int n, int *offset_ptr, int *dims, int max_dim, dtype ***in_loss);

//void Ftranspose_impl(const dtype* x, dtype* r, int dim0, int dim1, int size);

void set_impl(dtype **x, int* dims, int n, int max_dim, dtype val);

void set_col_impl(dtype* x, int dim0, int col, int size, dtype val);

void set_cols_impl(dtype* x, int dim0, int* col, int col_num, dtype* val);

void get_col_impl(const dtype* x, dtype* r, int dim0, int col, int size);

void get_cols_impl(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num);

void FLookup_impl(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num);

void DLookup_impl(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num);

void Fadd_col_impl(dtype* x, const dtype* y, int col, int dim0, int size);

void Favgpooling_impl(dtype** x, dtype* y, int n, int r, int s);

void Davgpooling_impl(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx);

void Fsumpooling_impl(dtype** x, dtype* y, int n, int r, int s);

void Dsumpooling_impl(const dtype* gy, int gy_size, int gx_size, dtype** gx);

void Fmaxpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index);

void Dmaxpooling_impl(const dtype* gy, dtype** gx, int* index, int dim);

void Fminpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index);

void Dminpooling_impl(const dtype* gy, dtype** gx, int* index, int dim);

void Fdropout_impl(dtype** x, dtype drop_rate, dtype* mask_val, int* mask, int dim, int count, dtype** y);

void Fdropout_impl(dtype** x, dtype drop_rate, int dim, int count, dtype** y);

void Ddropout_impl(dtype** in_loss, int* mask, int dim, int count, dtype** loss);

void Ddropout_impl(dtype** loss, int* mask, int dim, int count);
