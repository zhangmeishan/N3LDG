#include "kernel.cuh"


#define THREADS_PER_BLOCK 1024

__global__ void Fadd_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] + y[index];
	}
}

void Fadd_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fadd_inplace_kernel(dtype* x, const dtype* y, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		x[index] += y[index];
	}
}

void Fadd_inplace_impl(dtype * x, const dtype *y, int size) {
	Fadd_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, size);
	//cudaDeviceSynchronize();
}


__global__ void Dadd_inplace_kernel(dtype* in_loss, const dtype* loss, int dim, int n, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dim) {
		dtype sum = in_loss[index];
		for(int idx = 0; idx < n; idx++) {
			int i = idx * dim + index;
			sum += loss[i];
		}
		in_loss[index] = sum;
	}
}

void Dadd_inplace_impl(dtype* in_loss, const dtype* loss, int dim, int n) {
	int size = dim * n;
	Dadd_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(in_loss, loss, dim, n, size);
}

__global__ void Dadd_inplace_kernel(dtype** in_loss, const dtype* loss, int dim, int n, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim;
		int idy = index % dim;
		in_loss[idx][idy] += loss[index];
	}
}

void Dadd_inplace_impl(dtype** in_loss, const dtype* loss, int dim, int n) {
	int size = dim * n;
	Dadd_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(in_loss, loss, dim, n, size);
}

__global__ void Fadd_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = x[idx][idy] + y[idx][idy];
	}
}

void Fadd_impl(dtype** x, dtype** y,  dtype** r, int dim0, int size) {
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Fadd_inplace_kernel(dtype** x, dtype** y, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		x[idx][idy] += y[idx][idy];
	}
}

void Fadd_inplace_impl(dtype** x, dtype** y, int dim0, int size) {
	Fadd_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, dim0, size);
	//cudaDeviceSynchronize();
}

//cudaError_t res = cudaGetLastError();
//std::cout << cudaGetErrorString(res) << std::endl;

__global__ void Fadd_kernel(dtype*** x, dtype** y, int n, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int y_idx = index / dim0;
		int y_idy = index % dim0;
		y[y_idx][y_idy] = 0;
		int x_idx = y_idx;
		int x_idz = y_idy;
		for(int i = 0; i < n; i++) {
			y[y_idx][y_idy] += x[x_idx][i][x_idz];
		}
	}
}

void Fadd_impl(dtype*** x, dtype** y, int count, int n, int dim0) {
	int size = count * dim0;
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, n, dim0, size);

	//cudaError_t res = cudaGetLastError();
	//std::cout << cudaGetErrorString(res) << std::endl;
}


__global__ void Dadd_kernel(dtype*** in_loss, dtype** loss, int n, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int y_idx = index / dim0;
		int y_idy = index % dim0;
		int x_idx = y_idx;
		int x_idz = y_idy;
		for(int i = 0; i < n; i++) {
			in_loss[x_idx][i][x_idz] += loss[y_idx][y_idy];
		}
	}
}

void Dadd_impl(dtype*** in_loss, dtype** loss, int count, int n, int dim0) {
	int size = count * dim0;
	Dadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(in_loss, loss, n, dim0, size);

	//cudaError_t res = cudaGetLastError();
	//std::cout << cudaGetErrorString(res) << std::endl;
}

__global__ void Fsubtract_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] - y[index];
	}
}

void Fsubtract_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fsubtract_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fsubtract_inplace_kernel(dtype* x, const dtype* y, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		x[index] -= y[index];
	}
}

void Fsubtract_inplace_impl(dtype* x, const dtype* y, int size) {
	Fsubtract_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * y[index];
	}
}

void Fmultiply_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fmultiply_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_inplace_kernel(dtype* x, const dtype* y, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		x[index] *= y[index];
	}
}

void Fmultiply_inplace_impl(dtype* x, const dtype* y, int size) {
	Fmultiply_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, size);
	//cudaDeviceSynchronize();
}


__global__ void Fmultiply_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = x[idx][idy] * y[idx][idy];
	}
}

void Fmultiply_impl(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	Fmultiply_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_inplace_kernel(dtype** x, dtype** y, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		x[idx][idy] *= y[idx][idy];
	}
}

void Fmultiply_inplace_impl(dtype** x, dtype** y, int dim0, int size) {
	Fmultiply_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Fdivide_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] / y[index];
	}
}

void Fdivide_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fdivide_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_scalar_kernel(const dtype* x, const dtype y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * y;
	}
}

void Fmultiply_scalar_impl(const dtype* x, const dtype y, dtype* r, int size) {
	Fmultiply_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_scalar_kernel(const dtype* x, const dtype* scalar, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * scalar[0];
	}
}

void Fmultiply_scalar_impl(const dtype* x, const dtype* scalar, dtype* r, int size) {
	Fmultiply_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, scalar, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fmultiply_scalar_inplace_kernel(dtype* x, const dtype y, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		x[index] *= y;
	}
}

void Fmultiply_scalar_inplace_impl(dtype* x, const dtype y, int size) {
	Fmultiply_scalar_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, size);
	//cudaDeviceSynchronize();
}

__global__ void Fadd_scalar_kernel(const dtype* x, const dtype y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] + y;
	}
}

void Fadd_scalar_impl(const dtype* x, const dtype y, dtype* r, int size) {
	Fadd_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fadd_scalar_inplace_kernel(dtype* x, const dtype y, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		x[index] += y;
	}
}

void Fadd_scalar_inplace_impl(dtype* x, const dtype y, int size) {
	Fadd_scalar_inplace_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, size);
	//cudaDeviceSynchronize();
}

__global__ void Ddot_kernel(dtype **val, dtype **loss, dtype **in_loss, int dim0, int n, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		//in_loss[idx][idy] += (loss[idx][0] * val[idx][idy]);
		atomicAdd(in_loss[idx] + idy, (loss[idx][0] * val[idx][idy]));

	}
}

void Ddot_impl(dtype **val, dtype **loss, dtype **in_loss, int dim0, int n) {
	int size = dim0 * n;
	Ddot_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(val, loss, in_loss, dim0, n, size);
}

__global__ void Fsquare_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * x[index];
	}
}

void Fsquare_impl(const dtype* x, dtype* r, int size) {
	Fsquare_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Ftanh_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = tanh(x[index]);
	}
}

void Ftanh_impl(const dtype* x, dtype* r, int size) {
	Ftanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Ftanh_kernel(dtype** x, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = tanh(x[idx][idy]);
	}
}

void Ftanh_impl(dtype** x, dtype** r, int dim0, int size) {
	Ftanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Fsigmoid_kernel(dtype** x, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = 1.0 / (1.0 + exp(-x[idx][idy]));
	}
}

void Fsigmoid_impl(dtype** x, dtype** r, int dim0, int size) {
	Fsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Dsigmoid_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = (1 - y[idx][idy]) * y[idx][idy];
	}
}

void Dsigmoid_impl(dtype** x, dtype** y, dtype** r, int dim0, int size){
	Dsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	//cudaDeviceSynchronize();
}


__global__ void Dtanh_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = (1 + y[index]) * (1 - y[index]);
	}
}

void Dtanh_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Dtanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Dtanh_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = (1 + y[idx][idy]) * (1 - y[idx][idy]);
	}
}

void Dtanh_impl(dtype** x, dtype** y, dtype** r, int dim0, int size){
	Dtanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	//cudaDeviceSynchronize();
}

__global__ void Fsigmoid_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = 1.0 / (1.0 + exp(-x[index]));
	}
}

void Fsigmoid_impl(const dtype* x, dtype* r, int size) {
	Fsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Dsigmoid_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = (1 - y[index]) * y[index];
	}
}

void Dsigmoid_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Dsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	//cudaDeviceSynchronize();
}

__global__ void Fsqrt_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = sqrt(x[index]);
	}
}

void Fsqrt_impl(const dtype* x, dtype* r, int size) {
	Fsqrt_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	//cudaDeviceSynchronize();
}

__global__ void concat_kernel(dtype **src, dtype* dst, int src_dim, int dst_dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dst_dim) {
		int idx = index / src_dim;
		int idy = index % src_dim;
		dst[index] = src[idx][idy];
	}
}

void concat_impl(dtype **src, dtype* dst, int src_dim, int dst_dim) {
	concat_kernel<<<(dst_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, dst, src_dim, dst_dim);
	//cudaDeviceSynchronize();
}

__global__ void concat_kernel(dtype ***src, int* dims, int max_dim, int count, int n, int size, dtype **dst) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int src_idx = index / (max_dim * n);
		int src_idy = index % (max_dim * n) / max_dim;
		int src_idz = index % (max_dim * n) % max_dim;

		
		if(src_idz < dims[src_idx]) {
			int dst_idx = src_idx;
			int dst_idy = src_idy * dims[src_idx] + src_idz;
			dst[dst_idx][dst_idy] = src[src_idx][src_idy][src_idz];
		}
	}
}

void concat_impl(dtype ***src, int* dims, int max_dim, int count, int n, dtype **dst) {
	int size = max_dim * count * n;
	concat_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, dims, max_dim, count, n, size, dst);
	//cudaDeviceSynchronize();
}

__global__ void unconcat_kernel(const dtype *src, dtype **dst, int src_dim, int dst_dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < src_dim) {
		int idx = index / dst_dim;
		int idy = index % dst_dim;
		dst[idx][idy] = src[index];
	}
}

void unconcat_impl(const dtype *src, dtype** dst, int src_dim, int dst_dim) {
	unconcat_kernel<<<(src_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, dst, src_dim, dst_dim);
	//cudaDeviceSynchronize();
}


__global__ void Fconcat_kernel(dtype ***src, int count, int n, int *offset_ptr, int *dims, int max_dim, int dst_size, dtype **dst) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dst_size) {
		int dst_idx = index / (n * max_dim);

		int src_idx = dst_idx;
		int src_idy = index % (n * max_dim) / max_dim;
		int src_idz = index % (n * max_dim) % max_dim;
		//printf("%d, %d, %d, %d\n", index, src_idx, src_idy, src_idz);
		if(src_idz < dims[src_idy]) {

			int dst_idy = offset_ptr[src_idy] + src_idz;
			dst[dst_idx][dst_idy] = src[src_idx][src_idy][src_idz];
			//printf("%d, %d, %d, %d\n", index, src_idx, src_idy, src_idz);
		}

		//printf("%d\n", offset_ptr[index]);
	}
}

void Fconcat_impl(dtype ***src, int count, int n, int *offset_ptr, int *dims, int max_dim, dtype **dst) {
	int dst_size = count * n * max_dim;
	Fconcat_kernel<<<(dst_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, count, n, offset_ptr, dims, max_dim, dst_size, dst);
	//cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
	//cudaDeviceSynchronize();
}

__global__ void Dconcat_kernel(dtype **loss, int count, int n, int *offset_ptr, int *dims, int max_dim, int loss_size, dtype ***in_loss) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < loss_size) {
		int dst_idx = index / (n * max_dim);

		int src_idx = dst_idx;
		int src_idy = index % (n * max_dim) / max_dim;
		int src_idz = index % (n * max_dim) % max_dim;
		if(src_idz < dims[src_idy]) {
			int dst_idy = offset_ptr[src_idy] + src_idz;
			atomicAdd(in_loss[src_idx][src_idy] + src_idz, loss[dst_idx][dst_idy]);
			//printf("%d, %d, %d, %d\n", index, src_idx, src_idy, src_idz);
		}
		//in_loss[src_idx][src_idy][src_idz] += loss[dst_idx][dst_idy];
	}
}

void Dconcat_impl(dtype **loss, int count, int n, int *offset_ptr, int *dims, int max_dim, dtype ***in_loss) {
	int loss_size = count * n * max_dim;
	Dconcat_kernel<<<(loss_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(loss, count, n, offset_ptr, dims, max_dim, loss_size, in_loss);
	//cudaDeviceSynchronize();
}

/*
__global__ void Ftranspose_kernel(const dtype* x, dtype* r, int dim0, int dim1, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index % dim0 * dim1 + index / dim0];
	}
}

void Ftranspose_impl(const dtype* x, dtype* r, int dim0, int dim1, int size) {
	Ftranspose_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, dim1, size);
	//cudaDeviceSynchronize();
}

*/
__global__ void set_kernel(dtype **x, int* dims, int n, int max_dim, dtype val) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = index / max_dim;
	int idy = index % max_dim;
	if (idx < n && idy < dims[idx]) {
		x[idx][idy] = val;
	}
}

void set_impl(dtype **x, int* dims, int n, int max_dim, dtype val) {
	int size = n * max_dim;
	set_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dims, n, max_dim, val);
	//cudaDeviceSynchronize();
}

__global__ void set_col_kernel(dtype* x, int dim0, int col, int size, dtype val) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		x[i] = val;
	}
}

void set_col_impl(dtype* x, int dim0, int col, int size, dtype val) {
	set_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dim0, col, size, val);
	//cudaDeviceSynchronize();
}

__global__ void set_cols_kernel(dtype* x, int dim0, int* cols, int col_num, dtype* val, int val_size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < val_size) {
		int col_num = cols[index / dim0];
		int offset = index % dim0;
		x[col_num * dim0 + offset] = val[index];
	}
}


void set_cols_impl(dtype* x, int dim0, int* cols, int col_num, dtype* val) {
	int val_size = col_num * dim0;
	set_cols_kernel<<<(val_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dim0, cols, col_num, val, val_size);
	//cudaDeviceSynchronize();
}

__global__ void FLookup_kernel(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < r_size) {
		int col_index = index / xdim0;
		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % xdim0;
			int x_index = col * xdim0 + offset;
			if(x_index < xdim0 * xdim1) {
			   	r[col_index][offset] = x[x_index];
			}
		}
	}
}

void FLookup_impl(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	FLookup_kernel<<<(r_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(x, r, xdim0, xdim1, r_size, cols, col_num);	
	//cudaDeviceSynchronize();
}

__global__ void DLookup_kernel(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < l_size) {
		int col_index = index / gxdim0;
		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % gxdim0;
			int gx_index = col * gxdim0 + offset;
			if(gx_index < gxdim0 * gxdim1) {
				atomicAdd(gx + gx_index, loss[col_index][offset]);
			   	//gx[gx_index] += loss[col_index][offset];
			}
		}
	}
}

void DLookup_impl(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num) {
	DLookup_kernel<<<(l_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(gx, loss, gxdim0, gxdim1, l_size, cols, col_num);	
	//cudaDeviceSynchronize();
}

__global__ void get_cols_kernel(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < r_size) {
		int col_index = index / xdim0;

		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % xdim0;
			int x_index = col * xdim0 + offset;
			if(x_index < xdim0 * xdim1) {
			   	r[index] = x[x_index];
			}
		}
	}
}

void get_cols_impl(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	get_cols_kernel<<<(r_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(x, r, xdim0, xdim1, r_size, cols, col_num);	
	//cudaDeviceSynchronize();
}

__global__ void get_col_kernel(const dtype* x, dtype* r, int dim0, int col, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		r[index] = x[i];
	}
}

void get_col_impl(const dtype* x, dtype* r, int dim0, int col, int size) {
	get_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, col, size);
	//cudaDeviceSynchronize();
}

__global__ void Fadd_col_kernel(dtype* x, const dtype* y, int col, int dim0, int size){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		x[i] = x[i] + y[index];
	}
}

void Fadd_col_impl(dtype* x, const dtype* y, int col, int dim0, int size) {
	Fadd_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, col, dim0, size);
	//cudaDeviceSynchronize();
}


template<int BLOCK_SIZE>
__global__ void Favgpooling_kernel(
		dtype **px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	int index_start = bid % skip + (bid / skip) * skip * n;
	temp[tid] = 0;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		temp[tid] += px[idx][idy];
	}
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] += temp[tid + k]; \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) py[bid] = temp[0] / n;
}

void Favgpooling_impl(dtype** x, dtype* y, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Favgpooling_kernel<k><<<r, k>>>(x, s, n, y); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	//cudaDeviceSynchronize();
}

__global__ void Davgpooling_kernel(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < gx_size) {
		int idx = i / gy_size;
		int idy = i % gy_size;
		atomicAdd(gx[idx] + idy, gy[idy] / n);
		//gx[idx][idy] += (gy[idy] / n);
	}
}

void Davgpooling_impl(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx) {
	Davgpooling_kernel<<<(gx_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gy_size, gx_size, n, gx);
	//cudaDeviceSynchronize();
}

template<int BLOCK_SIZE>
__global__ void Fsumpooling_kernel(
		dtype **px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	int index_start = bid % skip + (bid / skip) * skip * n;
	temp[tid] = 0;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		temp[tid] += val;
	}
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] += temp[tid + k]; \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) py[bid] = temp[0];
}

void Fsumpooling_impl(dtype** x, dtype* y, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fsumpooling_kernel<k><<<r, k>>>(x, s, n, y); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	//cudaDeviceSynchronize();	
}

__global__ void Dsumpooling_kernel(const dtype* gy, int gy_size, int gx_size, dtype** gx) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < gx_size) {
		int idx = i / gy_size;
		int idy = i % gy_size;
		atomicAdd(gx[idx] + idy, gy[idy]);
		//gx[idx][idy] += gy[idy];
	}
}

void Dsumpooling_impl(const dtype* gy, int gy_size, int gx_size, dtype** gx) {
	Dsumpooling_kernel<<<(gx_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gy_size, gx_size, gx);
	//cudaDeviceSynchronize();	
}

template<int BLOCK_SIZE>
__global__ void Fmaxpooling_kernel(
		dtype **px, int skip, int n, dtype *py, int* index) {
	__shared__ dtype temp[BLOCK_SIZE];
	__shared__ int temp_index[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	dtype thread_max = NEGATIVE_INFINITY;
	int index_start = bid % skip + (bid / skip) * skip * n;
	int index_max;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		if(val > thread_max) {
			thread_max = val;
			index_max = index_start + i * skip;
		}
	}
	temp[tid] = thread_max;
	temp_index[tid] = index_max;
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) if(temp[tid + k] > temp[tid]) {temp[tid] = temp[tid + k]; temp_index[tid] = temp_index[tid + k];} \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) {py[bid] = temp[0]; index[bid] = temp_index[0];}
}

void Fmaxpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index){
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fmaxpooling_kernel<k><<<r, k>>>(x, s, n, y, index); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	//cudaDeviceSynchronize();	
}

__global__ void Dmaxpooling_kernel(const dtype* gy, dtype** gx, int* index, int dim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < dim) {
		int idx = index[i] / dim;
		int idy = index[i] % dim;

		atomicAdd(gx[idx] + idy, gy[i]);
		//gx[idx][idy] += gy[i];
	}
}

void Dmaxpooling_impl(const dtype* gy, dtype** gx, int* index, int dim) {
	Dmaxpooling_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gx, index, dim);
	
	//cudaDeviceSynchronize();	
}

template<int BLOCK_SIZE>
__global__ void Fminpooling_kernel(
		dtype **px, int skip, int n, dtype *py, int* index) {
	__shared__ dtype temp[BLOCK_SIZE];
	__shared__ int temp_index[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	dtype thread_min = POSITIVE_INFINITY;
	int index_start = bid % skip + (bid / skip) * skip * n;
	int index_min;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		if(val <  thread_min) {
			thread_min = val;
			index_min = index_start + i * skip;
		}
	}
	temp[tid] = thread_min;
	temp_index[tid] = index_min;
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) if(temp[tid + k] < temp[tid]) {temp[tid] = temp[tid + k]; temp_index[tid] = temp_index[tid + k];} \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) {py[bid] = temp[0]; index[bid] = temp_index[0];}
}

void Fminpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fminpooling_kernel<k><<<r, k>>>(x, s, n, y, index); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	//cudaDeviceSynchronize();	
}

__global__ void Dminpooling_kernel(const dtype* gy, dtype** gx, int* index, int dim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < dim) {
		int idx = index[i] / dim;
		int idy = index[i] % dim;
		atomicAdd(gx[idx] + idy, gy[i]);
		//gx[idx][idy] += gy[i];
	}
}

void Dminpooling_impl(const dtype* gy, dtype** gx, int* index, int dim) {
	Dminpooling_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gx, index, dim);
	//cudaDeviceSynchronize();	
}

__global__ void Fdropout_kernel(dtype** x, dtype drop_rate, dtype* mask_val, int* mask, int dim, int size, dtype** y) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		mask[index] = mask_val[index] > drop_rate ? 1 : 0;

		int idx = index / dim;	
		int idy = index % dim;

		y[idx][idy] = x[idx][idy] * mask[index];
	}
}

void Fdropout_impl(dtype** x, dtype drop_rate, dtype* mask_val, int* mask, int dim, int count, dtype** y) {
	int size = dim * count;
	Fdropout_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, drop_rate, mask_val, mask, dim, size, y);
}

__global__ void Ddropout_kernel(dtype** in_loss, int* mask, int dim, int size, dtype** loss) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim;	
		int idy = index % dim;
		in_loss[idx][idy] += loss[idx][idy] * mask[index];
	}
}

void Ddropout_impl(dtype** in_loss, int* mask, int dim, int count, dtype** loss) {
	int size = dim * count;
	Ddropout_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(in_loss, mask, dim, size, loss);
}

__global__ void Fdropout_kernel(dtype** x, dtype drop_rate, int dim, int size, dtype** y) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {

		int idx = index / dim;	
		int idy = index % dim;

		y[idx][idy] = x[idx][idy] * (1 - drop_rate);
	}
}

void Fdropout_impl(dtype** x, dtype drop_rate, int dim, int count, dtype** y) {
	int size = dim * count;
	Fdropout_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, drop_rate, dim, size, y);
}

__global__ void Ddropout_kernel(dtype** loss, int* mask, int dim, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim;	
		int idy = index % dim;
		loss[idx][idy] *= mask[index];
	}
}

void Ddropout_impl(dtype** loss, int* mask, int dim, int count) {
	int size = dim * count;
	Ddropout_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(loss, mask, dim, size);
}
