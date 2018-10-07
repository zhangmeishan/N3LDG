#ifndef CUDA_DEVICE
#define CUDA_DEVICE

#include <iostream>
#include <iomanip>
#include <chrono>

#include "memory_pool.h"
#include "Device.h"
#include "MyLib.h"
#include "kernel.cuh"

static void *my_alloc(std::size_t size) {
	void *ptr;
	cudaMalloc((void **)&ptr, size);
	//cudaHostAlloc((void **)&ptr, size, cudaHostAllocDefault);
	return ptr;
}

static void my_delete(void *ptr) {
	//cudaFreeHost(ptr);
	cudaFree(ptr);
}

MemoryPool* get_mem_pool() {
	 static MemoryPool mem_pool(my_alloc, my_delete);
	 return &mem_pool;
}


#define MEM_POOL get_mem_pool()


// return 2d array and 3d array.
// only for cuda device.
class ArrayPtr {
	private:
		std::shared_ptr<void> ptr_;
		vector<std::shared_ptr<void> > vec_tmp_ptr; 
	
	public:

	ArrayPtr(const vector<LDG::PTensor>& vec) {
		int n = vec.size();
		int mem_size = sizeof(dtype*) * n;

		vector<const dtype*> vec_ptr(n);
		for(int i = 0; i < n; i++) {
			vec_ptr[i] = static_cast<const dtype* >(vec[i]->get_handle());
		}
		ptr_ = MEM_POOL->allocate(mem_size); 
		const dtype **v_data = static_cast<const dtype **>(ptr_.get());
		//cudaMemcpy(v_data, (dtype**)vec_ptr.data(), mem_size, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(v_data, (dtype**)vec_ptr.data(), mem_size, cudaMemcpyHostToDevice, 0);
	}	


	ArrayPtr(const vector<vector<LDG::PTensor> >& vec_vec) {
		int count = vec_vec.size();
		int n = vec_vec[0].size();
		int mem_size = sizeof(dtype**) * count;
		ptr_ = MEM_POOL->allocate(mem_size); 
		dtype ***v_v_data = static_cast<dtype ***>(ptr_.get());

		vector<dtype**> vec_vec_ptr(count);
		vec_tmp_ptr.resize(count); 
		for (int idx = 0; idx < count; idx++) {
			const vector<LDG::PTensor> vec = vec_vec[idx];
			int mem_piece_size = sizeof(dtype*) * n;
			vector<const dtype*> vec_ptr(n);
			for(int i = 0; i < n; i++) {
				vec_ptr[i] = static_cast<const dtype* >(vec[i]->get_handle());
			}

			vec_tmp_ptr[idx] = MEM_POOL->allocate(mem_piece_size); 
			dtype **v_data = static_cast<dtype **>(vec_tmp_ptr[idx].get());
			//cudaMemcpy(v_data, (dtype**)vec_ptr.data(), mem_piece_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(v_data, (dtype**)vec_ptr.data(), mem_piece_size, cudaMemcpyHostToDevice, 0);

			vec_vec_ptr[idx] = v_data;
			//concat_impl(vec_vec_ptr_x[idx] , MDATA(*vec_r[idx]), src_dim, src_dim * n);
		}
		//cudaMemcpy(v_v_data, (dtype***)vec_vec_ptr.data(), mem_size, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(v_v_data, (dtype***)vec_vec_ptr.data(), mem_size, cudaMemcpyHostToDevice, 0);
	}

	void *get_ptr() {
		return ptr_.get();
	}
};

class CudaDevice : public Device {
	private:
		cublasHandle_t handle;
		curandGenerator_t rndGen;
		static const int STREAM_NUM = 128;
		cudaStream_t streams[STREAM_NUM];

	public:

	public:
		CudaDevice() {
			Device::device_type = CUDA;
			cublasCreate(&handle);
			curandCreateGenerator(&rndGen, CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(rndGen, 0); //seed
			for (int i = 0; i < STREAM_NUM; i++) {
				cudaStreamCreate(&streams[i]);
			}
		}

		~CudaDevice() {
			cublasDestroy(handle);
			curandDestroyGenerator(rndGen);
			for (int i = 0; i < STREAM_NUM; i++) {
				cudaStreamDestroy(streams[i]);
			}
		}

	public:

		void init_index_ptr(IndexPtr& index_ptr, int size) {
			int mem_size = sizeof(int) * size;
			index_ptr.ptr_ = MEM_POOL->allocate(mem_size);
			index_ptr.size = size;	
		}

		void init_dtype_ptr(DtypePtr& dtype_ptr, int size) {
			int mem_size = sizeof(dtype) * size;
			dtype_ptr.ptr_ = MEM_POOL->allocate(mem_size);
			dtype_ptr.size = size;	
		}

		void init(LDG::Tensor &t, const Shape &shape) {
			malloc(t, shape);
			zero(t);
		}

		void init_index_ptr(std::shared_ptr<void>& ptr_, int mem_size) {
			ptr_ = MEM_POOL->allocate(mem_size);
		}
		/*
		void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) {
			if(src.shape().has_same_dims(dst.shape())
					&& src.device_type == dst.device_type) {
				int memsize = src.shape().size() * sizeof(dtype);
				cudaMemcpy(dst.v, src.v, memsize, cudaMemcpyDeviceToDevice);
			} else
				cout << "copy error"  << endl;
		}
		*/

		void set(vector<LDG::PTensor> &vec_t, dtype val) {
			int n = vec_t.size();

			int max_dim = -1;
			vector<int> vec_dims;
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_t[idx]->shape()[0];
				vec_dims.push_back(dim);
				if (dim > max_dim)
					max_dim = dim;
			}
			int mem_size = sizeof(int) * n;
			IndexPtr dims_ptr;
			init_index_ptr(dims_ptr, n);
			cudaMemcpyAsync(dims_ptr.get_ptr(), (int*)vec_dims.data(), mem_size, cudaMemcpyHostToDevice, 0);

			ArrayPtr t_ptr(vec_t);
			dtype **t = (dtype**)t_ptr.get_ptr();

			set_impl(t, dims_ptr.get_ptr(), n, max_dim, val);
		}



		void set(LDG::Tensor &t, const dtype* host_data, int h_size) {
			int size = t.shape().size();
			if (size == h_size) {
				int memsize = sizeof(dtype) * size;
				//cudaMemcpy(MDATA(t), host_data, memsize, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(MDATA(t), host_data, memsize, cudaMemcpyHostToDevice, 0);
			} else
				cout << "set size not match" << endl;
		}

		void set_col(LDG::Tensor &t, int col, dtype val) {
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];
			if(col < dim1) {
				set_col_impl(MDATA(t), dim0, col, t.shape().size(), val);
			} else
				std::cout << "set col beyond dim1 " << endl;
		}

		void set_cols(LDG::Tensor &t, const vector<int>& cols, const vector<dtype>& vals){
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];

			int col_num = cols.size();

			assert(dim0 * col_num == vals.size());
			for(int idx = 0; idx < col_num; idx++)
				assert(cols[idx] < dim1);

			int mem_size = sizeof(int) * col_num;
			IndexPtr gpu_cols;
			init_index_ptr(gpu_cols, col_num);
			cudaMemcpyAsync(gpu_cols.get_ptr(), cols.data(), mem_size, cudaMemcpyHostToDevice, 0);

			
			mem_size = sizeof(dtype) * vals.size();
			DtypePtr gpu_vals;
			init_dtype_ptr(gpu_vals, vals.size());
			cudaMemcpyAsync(gpu_vals.get_ptr(), vals.data(), mem_size, cudaMemcpyHostToDevice, 0);

			set_cols_impl(MDATA(t), dim0, gpu_cols.get_ptr(), col_num, gpu_vals.get_ptr());
		}

		void malloc(LDG::Tensor &t, const Shape &shape) {
			t.device_type = CUDA;
			t.shape_ = shape;
			int size = shape.size();
			int mem_size = sizeof(dtype) * size;

			t.handle_ = MEM_POOL->allocate(mem_size);
		}

		void zero(LDG::Tensor &t) {
			set(t, 0);
		}

		void set(LDG::Tensor &t, dtype val) {
			int size = t.shape().size();
			vector<dtype> zero_host(size);
			for (int idx = 0; idx < size; idx++)
				zero_host[idx] = val;
			set(t, zero_host.data(), size);
		}

		void set(LDG::Tensor &t, const vector<dtype>& vec_val) {
			int size = t.shape().size();
			if (vec_val.size() == size) {
				//cudaMemcpy(MDATA(t), vec_val.data(), sizeof(dtype) * size, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(MDATA(t), vec_val.data(), sizeof(dtype) * size, cudaMemcpyHostToDevice, 0);
			} else 
				cout << "set error dim is not match" << endl;
		}

		void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) {
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			malloc(r, Shape({dim0, 1}));

			if(col < dim1) {
				get_col_impl(CDATA(x), MDATA(r), dim0, col, x.shape().size());
			} else
				cout << "get col, col beyond" << endl;
		}

		void get_cols(const LDG::Tensor& x, int* cols, int col_num, LDG::Tensor& r) {
			int memsize = sizeof(int) * col_num;

			IndexPtr gpu_cols;
			init_index_ptr(gpu_cols, col_num);

			//cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_cols.get_ptr(), cols, memsize, cudaMemcpyHostToDevice, 0);

			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			malloc(r, Shape({xdim0 * col_num, 1}));

			int r_size = r.shape().size();

			for(int idx = 0; idx < col_num; idx++)
				assert(cols[idx] < xdim1);

			malloc(r, Shape({xdim0, col_num}));
			get_cols_impl(CDATA(x), MDATA(r), xdim0, xdim1, r_size, gpu_cols.get_ptr(), col_num);
		}

		void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_r) {
			int size = vec_cols.size();
			FLookup(x, vec_cols.data(), size, vec_r);
		}

		void DLookup(LDG::Tensor& gx, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_loss) {
			int size = vec_cols.size();
			DLookup(gx, vec_cols.data(), size, vec_loss);
		}

		void FLookup(const LDG::Tensor& x, const int* cols, int col_num, vector<LDG::PTensor>& vec_r) {
			if(vec_r.size() != col_num)
				cout << "error vec_r size is not matched." << endl;

			int n = col_num;
			int mem_size = sizeof(int) * n;
			IndexPtr gpu_cols;
			init_index_ptr(gpu_cols, n);
			//cudaMemcpy(gpu_cols, cols, mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_cols.get_ptr(), cols, mem_size, cudaMemcpyHostToDevice, 0);

			ArrayPtr r_ptr(vec_r);
			dtype **r = (dtype**)r_ptr.get_ptr();
			
			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int rdim0 = vec_r[0]->shape()[0];
			int r_size = rdim0 * col_num;
			
			if(xdim0 == rdim0) {
				FLookup_impl(CDATA(x), r, xdim0, xdim1, r_size, gpu_cols.get_ptr(), col_num);
			} else
				cout << "get col dims are not matched" << endl;
		}


		void DLookup(LDG::Tensor& gx, const int* cols, int col_num, vector<LDG::PTensor>& vec_loss) {
			if(vec_loss.size() != col_num) {
				cout << "error vec_loss size is not matched." << endl;
			}

			int n = col_num;
			int memsize = sizeof(int) * col_num;
			IndexPtr gpu_cols;
			init_index_ptr(gpu_cols, n);

			//cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(gpu_cols.get_ptr(), cols, memsize, cudaMemcpyHostToDevice, 0);


			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			int gxdim0 = gx.shape()[0];
			int gxdim1 = gx.shape()[1];
			int ldim0 = vec_loss[0]->shape()[0];
			int l_size = ldim0 * col_num;
			
			if(gxdim0 == ldim0) {
				DLookup_impl(MDATA(gx), loss, gxdim0, gxdim1, l_size, gpu_cols.get_ptr(), col_num);
			} else
				cout << "get col dims are not matched" << endl;
		}

		void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {
			int size = shape.size();
			int memsize = sizeof(dtype) * size;
			vector<dtype> host_data(size);
			dtype min = lower, max = upper;
			for (int i = 0; i < size; i++) {
				host_data[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
			}

			malloc(t, shape);
			cudaMemcpyAsync(MDATA(t), host_data.data(), memsize, cudaMemcpyHostToDevice, 0);
		} 

		void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p){}

		void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}

		void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}


		void Fequal(const LDG::Tensor& x, LDG::Tensor& r){}

		void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Ftanh_impl(CDATA(x), MDATA(r), x_size);
		}

		void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				int n = n_x;
				for(int idx = 0; idx < n; idx++)
					malloc(*vec_r[idx], vec_x[idx]->shape());

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();


				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Ftanh_impl(x, r, dim0, size);
			} else 
				std::cout << "error the number of Ftanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				int n = n_r;
				for(int idx = 0; idx < n_r; idx++)
					malloc(*vec_r[idx], vec_x[idx]->shape());

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Fsigmoid_impl(x, r, dim0, size);
			} else 
				std::cout << "error the number of Fsigmoid tensors is not matched "<<endl;
		}

		void Dsigmoid(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {

				for(int idx = 0; idx < n_x; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}
				int n = n_x;

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();


				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dsigmoid_impl(x, y, r, dim0, size);
			} else 
				std::cout << "error the number of Dsigmoid tensors is not matched "<<endl;
		}


		void Dtanh(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {

				for(int idx = 0; idx < n_x; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}
				int n = n_x;

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dtanh_impl(x, y, r, dim0, size);
			} else 
				std::cout << "error the number of Dtanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsigmoid_impl(CDATA(x), MDATA(r), x_size);
		}

		void Frelu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fexp(const LDG::Tensor& x, LDG::Tensor& r){}
		void Flog(const LDG::Tensor& x, LDG::Tensor& r){}


		void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsquare_impl(CDATA(x), MDATA(r), x_size);
		}

		void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsqrt_impl(CDATA(x), MDATA(r), x_size);
		}

		void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			malloc(r, x.shape());
			if(x_size == y_size)
				Dtanh_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			else
				std::cout << "error, dtanh dim is not match" << std::endl;
		}

		void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			malloc(r, x.shape());
			if(x_size == y_size)
				Dsigmoid_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			else
				std::cout << "error, dsigmoid dim is not match" << std::endl;
		}

		void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		/*
		virtual LDG::Tensor Fadd(const LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			LDG::Tensor r;
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fadd_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, add dim is not match" << std::endl;
			return r;
		}
		*/

		void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fadd_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, add dim is not match" << std::endl;
		}

		virtual void Fadd_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fadd_inplace_impl(MDATA(x), CDATA(y), x_size);
			} else 
				std::cout << "error, add dim is not match" << std::endl;
		}

		void Dadd_inplace(vector<LDG::PTensor>& vec_in_loss, const LDG::Tensor& loss) {
			int n = vec_in_loss.size();
			int dim = loss.shape()[0];
			assert(n == loss.shape()[1]);
			assert(dim == vec_in_loss[0]->shape()[0]);
			
			ArrayPtr in_loss_ptr(vec_in_loss);
			dtype **in_loss = (dtype**)in_loss_ptr.get_ptr();

			Dadd_inplace_impl(in_loss, CDATA(loss), dim, n);

		}

		void Dadd_inplace(LDG::Tensor& in_loss, const LDG::Tensor& loss) {
			const int dim = loss.shape()[0];
			assert(dim == loss.shape()[0]);
			const int n = loss.shape()[1];
			Dadd_inplace_impl(MDATA(in_loss), CDATA(loss), dim, n);
		}

		/*
		virtual void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r) {
			int n = vec_y.size();
			int x_size = x.shape().size();
			malloc(r, x.shape());
			for(int idx = 0; idx < n; idx++){
				assert(x_size == vec_y[idx]->shape().size());
			}


			ArrayPtr y_ptr(vec_y);
			dtype **y = (dtype**)y_ptr.get_ptr();

			Fadd_impl(CDATA(x), y, MDATA(r), n, x_size);
		}
		*/

		void Fadd(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();

				int size = vec_x[0]->shape().size();
				Fadd_impl(x, y, r, size, size * n);
			} else {
				cout << "Fadd size is not matched" << endl;
			}
		}

		void Fadd_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
				}

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				int size = vec_x[0]->shape().size();
				Fadd_inplace_impl(x, y, size, size * n);
			} else {
				cout << "Fadd size is not matched" << endl;
			}
		}

		void Fadd(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_y) {
			int count = vec_vec_x.size();
			int n = vec_vec_x[0].size();
			assert(count == vec_y.size());
			
			for(int idx = 1; idx < count; idx++)
				assert(vec_vec_x[idx].size() == n);
			int y_dim = vec_y[0]->shape()[0];
			for(int idx = 0; idx < count; idx++) {
				assert(vec_y[idx]->shape()[0] == y_dim);
				for(int idy = 0; idy < n; idy++) {
					assert(y_dim == vec_vec_x[idx][idy]->shape()[0]);
				}
			}
			ArrayPtr x_ptr(vec_vec_x);
			dtype ***x = (dtype ***)x_ptr.get_ptr();

			ArrayPtr y_ptr(vec_y);
			dtype **y = (dtype **)y_ptr.get_ptr();

			Fadd_impl(x, y, count, n, y_dim);
		}

		void Dadd(vector<vector<LDG::PTensor> >& vec_vec_in_loss, const vector<LDG::PTensor>& vec_loss) {
			int count = vec_vec_in_loss.size();
			int n = vec_vec_in_loss[0].size();
			assert(count == vec_loss.size());
			
			for(int idx = 1; idx < count; idx++)
				assert(vec_vec_in_loss[idx].size() == n);
			int loss_dim = vec_loss[0]->shape()[0];
			for(int idx = 0; idx < count; idx++) {
				assert(vec_loss[idx]->shape()[0] == loss_dim);
				for(int idy = 0; idy < n; idy++) {
					assert(loss_dim == vec_vec_in_loss[idx][idy]->shape()[0]);
				}
			}
			ArrayPtr in_loss_ptr(vec_vec_in_loss);
			dtype ***in_loss = (dtype ***)in_loss_ptr.get_ptr();

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype **)loss_ptr.get_ptr();

			Dadd_impl(in_loss, loss, count, n, loss_dim);
		}

		void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fsubtract_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, subtract dim is not match" << std::endl;
		}

		void Fsubtract_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fsubtract_inplace_impl(MDATA(x), CDATA(y), x_size);
			}
			else
				std::cout << "error, subtract dim is not match" << std::endl;
		}

		void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fmultiply_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, multiply dim is not match" << std::endl;

		}

		void Fmultiply_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fmultiply_inplace_impl(MDATA(x), CDATA(y), x_size);
			}
			else
				std::cout << "error, multiply dim is not match" << std::endl;

		}

		void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fadd_scalar_impl(CDATA(x), y, MDATA(r), x_size);
		}

		void Fadd_scalar_inplace(LDG::Tensor& x, const dtype y) {
			int x_size = x.shape().size();
			Fadd_scalar_inplace_impl(MDATA(x), y, x_size);
		}


		void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fmultiply_scalar_impl(CDATA(x), y, MDATA(r), x_size);
		}

		void Fmultiply_scalar(const LDG::Tensor& x, const LDG::Tensor &scalar, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			assert(scalar.shape().size() == 1);
			Fmultiply_scalar_impl(CDATA(x), CDATA(scalar), MDATA(r), x_size);
		}

		void Fmultiply_scalar_inplace(LDG::Tensor& x, const dtype y) {
			int x_size = x.shape().size();
			Fmultiply_scalar_inplace_impl(MDATA(x), y, x_size);
		}


		void Fadd_col(LDG::Tensor& x, const LDG::Tensor& y_col, int col) {
			auto x_dims = x.shape().dims();
			auto y_dims = y_col.shape().dims();
			int size = x.shape().size();
			int dim0 = x_dims[0];
			int dim1 = x_dims[1];
			if(col >= dim1) {
				cout << "col index beyond x dim" << endl;	
				return;
			}

			if (y_dims[1] != 1) {
				cout << "y is not a vector" << endl;
				return;
			}
					
			Fadd_col_impl(MDATA(x), CDATA(y_col), col, dim0, size);
		}

		void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fdivide_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, divide dim is not match" << std::endl;
		}

		/*
		void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int x_size = vec_x.size();
			int y_size = vec_y.size();
			int x_dim = vec_x[0]->shape()[0];
			int y_dim = vec_y[0]->shape()[0];
			LDG::Tensor x, y;
			concat(vec_x, x);
			concat(vec_y, y);

			Fmatmul(x, y, r, tx, ty);
		} 
		*/
		
		/*
		void Dmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			LDG::Tensor x, y;
			int x_size = vec_x.size();
			int x_dim = vec_x[0]->shape()[0];
			concat(vec_x, x);

			int y_size = vec_y.size();
			int y_dim = vec_y[0]->shape()[0];
			concat(vec_y, y);

			Dmatmul(x, y, r, tx, ty);
		} 
		*/
		

		void Dmatmul(const LDG::Tensor& x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			LDG::Tensor y;
			int y_size = vec_y.size();
			int y_dim = vec_y[0]->shape()[0];
			concat(vec_y, y);

			Dmatmul(x, y, r, tx, ty);
		} 

		void Dmatmul(const LDG::Tensor& x, const LDG::Tensor &y, vector<LDG::PTensor> &vec_r,
				bool tx = false, bool ty = false) {
			LDG::Tensor r;
			int r_size = vec_r.size();
			int r_dim = vec_r[0]->shape()[0];

			malloc(r, Shape({r_dim, r_size}));
			Fmatmul(x, y, r, tx, ty);
			Dadd_inplace(vec_r, r);
		} 

		void Fdot(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r,
				bool tx = false, bool ty = false) {
			int x_size = vec_x.size();
			int y_size = vec_y.size();
			int r_size = vec_y.size();
			int x_dim = vec_x[0]->shape().size();
			int y_dim = vec_y[0]->shape().size();
			int r_dim = vec_r[0]->shape().size();
			LDG::Tensor x, y, r;
			malloc(r, Shape({r_dim, r_size}));
			concat(vec_x, x);
			concat(vec_y, y);

			int m = tx ? vec_x[0]->shape()[1] : vec_x[0]->shape()[0];
			int n = ty ? vec_y[0]->shape()[0] : vec_y[0]->shape()[1];
			int k = tx ?  vec_x[0]->shape()[0] : vec_x[0]->shape()[1];
			dtype alpha = 1;
			dtype beta =  0;

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? k : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? n : k;

#if USE_FLOAT
			cublasSgemmStridedBatched(handle, transx, transy,
					m, n, k,
					&alpha,
					CDATA(x), ldx,
					x_dim,
					CDATA(y), ldy,
					y_dim,
					&beta, 
					MDATA(r), m,
					r_dim,
					x_size);
#else                   
			cublasDgemmStridedBatched(handle, transx, transy,
					m, n, k,
					&alpha,
					CDATA(x), ldx,
					x_dim,
					CDATA(y), ldy,
					y_dim,
					&beta, 
					MDATA(r), m,
					r_dim,
					x_size);
#endif
			unconcat(r, vec_r);
		} 

		void Ddot(vector<LDG::PTensor> &vec_in_loss, const vector<LDG::PTensor> &vec_val, const vector<LDG::PTensor> &vec_loss) {
			int val_size = vec_val.size();
			int loss_size = vec_loss.size();
			int in_loss_size = vec_in_loss.size();
			assert(val_size == loss_size && val_size == in_loss_size);
			
			int dim0 = vec_val[0]->shape()[0];
			for(int idx = 0; idx < loss_size; idx++)
				assert(vec_loss[idx]->shape().size() == 1);

			ArrayPtr val_ptr(vec_val);
			dtype **val = (dtype**)val_ptr.get_ptr();

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			ArrayPtr in_loss_ptr(vec_in_loss);
			dtype **in_loss = (dtype**)in_loss_ptr.get_ptr();

			Ddot_impl(val, loss, in_loss, dim0, val_size);
		}

		void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = ty ? y.shape()[0] : y.shape()[1];
			int k = tx ?  x.shape()[0] : x.shape()[1];

			malloc(r, Shape({m, n}));

			dtype alpha = 1;
			dtype beta =  0;

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? k : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? n : k;

#if USE_FLOAT
			(cublasSgemm(handle, transx, transy, m, n, k,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#else                   
			(cublasDgemm(handle, transx, transy, m, n, k,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#endif                  
		}

		void Dmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = ty ? y.shape()[0] : y.shape()[1];
			int k = tx ?  x.shape()[0] : x.shape()[1];

			dtype alpha = 1;
			dtype beta =  1;

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? k : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? n : k;

#if USE_FLOAT
			(cublasSgemm(handle, transx, transy, m, n, k,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#else                   
			(cublasDgemm(handle, transx, transy, m, n, k,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#endif                  
		}

		void Dmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) {
			int size_y = vec_y.size();
			int size_r = vec_r.size();
			assert(size_y == size_r);

			LDG::Tensor y;
			int y_dim = vec_y[0]->shape()[0];
			concat(vec_y, y);

			LDG::Tensor r;
			int r_dim = vec_r[0]->shape()[0];
			concat(vec_r, r);

			Dmatmul(x, y, r, tx, ty);
			unconcat(r, vec_r);
		}

		void Fmultiply(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					int rdim0 = vec_r[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				ArrayPtr r_ptr(vec_r);
				dtype **r = (dtype**)r_ptr.get_ptr();
				int size = vec_x[0]->shape().size();

				Fmultiply_impl(x, y, r, size, size * n);
			} else {
				cout << "Fmultiply size is not matched" << endl;
			}
		}

		void Fmultiply_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
				}

				ArrayPtr x_ptr(vec_x);
				dtype **x = (dtype**)x_ptr.get_ptr();

				ArrayPtr y_ptr(vec_y);
				dtype **y = (dtype**)y_ptr.get_ptr();

				int size = vec_x[0]->shape().size();
				Fmultiply_inplace_impl(x, y, size, size * n);
			} else {
				cout << "Fmultiply size is not matched" << endl;
			}
		}
/*
		void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {
			int nDim = x.shape()[0];
			int memsize = nDim * sizeof(dtype);
			dtype x_host_data[nDim], r_host_data[nDim], scores[nDim];
			cudaMemcpy(x_host_data, x.v, memsize, cudaMemcpyDeviceToHost);

			int optLabel = -1;
			for (int i = 0; i < nDim; ++i) {
				if (optLabel < 0 || x_host_data[i] > x_host_data[optLabel])
					optLabel = i;
			}

			dtype sum2 = 0, maxScore = x_host_data[optLabel];
			for (int i = 0; i < nDim; ++i) {
				scores[i] = -1e10;
				scores[i] = exp(x_host_data[i] - maxScore);
				sum2 += scores[i];
			} 

			for (int i = 0; i < nDim; ++i) {
				r_host_data[i] = scores[i] / sum2;
			}
			cudaMemcpy(r.v, r_host_data,  memsize, cudaMemcpyHostToDevice);
		}

		void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
				LDG::Tensor& gx){
		}
		*/

		/*
		void concat(const vector<vector<LDG::PTensor> >& vec_vec_x, LDG::Tensor& r) {
			int max_dim = -1;
			int count = vec_vec_x.size();
			int n = vec_vec_x[0].size();
			vector<int> dim_offset(n), dims(n);
			int offset = 0;
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_vec_x[0][idx]->shape()[0];
				dim_offset[idx] = offset;
				dims[idx] = dim;
				if(max_dim < dim)
					max_dim = dim;
				offset += dim;
			}

			malloc(r, Shape({offset, count}));

			int mem_size = sizeof(int) * n;
			IndexPtr offset_ptr;
			init_index_ptr(offset_ptr, n);
			//cudaMemcpy(offset_ptr, (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(offset_ptr.get_ptr(), (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice, 0);

			IndexPtr dims_ptr;
			init_index_ptr(dims_ptr, n);
			//cudaMemcpy(dims_ptr, (int*)dims.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(dims_ptr.get_ptr(), (int*)dims.data(), mem_size, cudaMemcpyHostToDevice, 0);

			ArrayPtr src_ptr(vec_vec_x);
			dtype ***src = (dtype ***)src_ptr.get_ptr();

			concat_impl(src, count, n, offset_ptr.get_ptr(), dims_ptr.get_ptr(), offset, max_dim, MDATA(r));
		}
		*/

		void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {
			const Shape xshape = vec_x[0]->shape();
			int src_dim = xshape[0];
			int n = vec_x.size();
			int dst_dim = n * src_dim;

			malloc(r, Shape({src_dim, n}));

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();

			concat_impl(x, MDATA(r), src_dim, dst_dim);
		}

		void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
			const Shape xshape = vec_x[0]->shape();
			int src_dim = r.shape().size();
			int dst_dim = xshape[0];

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();

			unconcat_impl(CDATA(r), x, src_dim, dst_dim);
		}

		void Fconcat(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_r) {
			int count = vec_vec_x.size();
			int n = vec_vec_x[0].size();
			assert(count == vec_r.size());

			int max_dim = -1;
			vector<int> dim_offset(n), dims(n);
			int offset = 0;
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_vec_x[0][idx]->shape()[0];
				//cout << vec_vec_x[0][idx]->shape().to_string() << endl;
				dim_offset[idx] = offset;
				dims[idx] = dim;
				if(max_dim < dim)
					max_dim = dim;
				offset += dim;
			}

			int mem_size = sizeof(int) * n;
			IndexPtr offset_ptr;
			init_index_ptr(offset_ptr, n);
			//cudaMemcpy(offset_ptr, (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(offset_ptr.get_ptr(), (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice, 0);

			IndexPtr dims_ptr;
			init_index_ptr(dims_ptr, n);
			//cudaMemcpy(dims_ptr, (int*)dims.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(dims_ptr.get_ptr(), (int*)dims.data(), mem_size, cudaMemcpyHostToDevice, 0);

			ArrayPtr src_ptr(vec_vec_x);
			dtype ***src = (dtype ***)src_ptr.get_ptr();

			ArrayPtr dst_ptr(vec_r);
			dtype **dst = (dtype **)dst_ptr.get_ptr();

			int src_dim = vec_vec_x[0][0]->shape()[0];
			int src_size = src_dim * n * count ;
			int dst_dim = vec_r[0]->shape()[0];

			Fconcat_impl(src, count, n, offset_ptr.get_ptr(), dims_ptr.get_ptr(), max_dim, dst);
		}

		void Dconcat(vector<vector<LDG::PTensor> >& vec_vec_in_loss, const vector<LDG::PTensor>& vec_loss) {
			int n = vec_vec_in_loss[0].size();
			int count = vec_loss.size();
			assert(count == vec_vec_in_loss.size());

			int max_dim = -1;
			vector<int> dim_offset(n), dims(n);
			int offset = 0;
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_vec_in_loss[0][idx]->shape()[0];
				//cout << vec_vec_x[0][idx]->shape().to_string() << endl;
				dim_offset[idx] = offset;
				dims[idx] = dim;
				if(max_dim < dim)
					max_dim = dim;
				offset += dim;
			}

			int mem_size = sizeof(int) * n;
			IndexPtr offset_ptr;
			init_index_ptr(offset_ptr, n);
			//cudaMemcpy(offset_ptr, (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(offset_ptr.get_ptr(), (int*)dim_offset.data(), mem_size, cudaMemcpyHostToDevice, 0);

			IndexPtr dims_ptr;
			init_index_ptr(dims_ptr, n);
			//cudaMemcpy(dims_ptr, (int*)dims.data(), mem_size, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(dims_ptr.get_ptr(), (int*)dims.data(), mem_size, cudaMemcpyHostToDevice, 0);

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype **)loss_ptr.get_ptr();

			ArrayPtr in_loss_ptr(vec_vec_in_loss);
			dtype ***in_loss = (dtype ***)in_loss_ptr.get_ptr();

			Dconcat_impl(loss, count, n, offset_ptr.get_ptr(), dims_ptr.get_ptr(), max_dim, in_loss);
		}
		/*
		void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {
			if (gpu_tensor.device_type == CUDA && cpu_tensor.device_type == CPU) {
				if(gpu_tensor.shape().has_same_dims(cpu_tensor.shape())) {
					int memsize = gpu_tensor.shape().size() * sizeof(dtype);
					cudaMemcpy(cpu_tensor.v, gpu_tensor.v, memsize, cudaMemcpyDeviceToHost);		
				} else {
					cout << "gpu: " << gpu_tensor.shape().to_string() << " ";
					cout << "cpu: " << cpu_tensor.shape().to_string() << endl;
					cout << "to_cpu dims are not match." << endl;
				}
			} else {
				cout << "to_cpu tensor type is error" << endl;
			}
		}

		void to_gpu(const LDG::Tensor &cpu_tensor, LDG::Tensor& gpu_tensor) {
			if (gpu_tensor.device_type == CUDA && cpu_tensor.device_type == CPU) {
				if(gpu_tensor.shape().has_same_dims(cpu_tensor.shape())) {
					int memsize = cpu_tensor.shape().size() * sizeof(dtype);
					cudaMemcpy(gpu_tensor.v, cpu_tensor.v, memsize, cudaMemcpyHostToDevice);		
				} else {
					cout << "to_gpu dims are not match." << endl;
				}
			} else {
				cout << "to_cpu tensor type is error" << endl;
			}
		}
		*/

		/*
		void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			r.shape_ = Shape({dim1, dim0});
			if(x_size == r_size)
				Ftranspose_impl(x.v, r.v, dim0, dim1, x_size);
			else
				std::cout << "error, transpose dim is not match" << std::endl;
		}
		*/

		void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();

			Favgpooling_impl(x, MDATA(y), n, r, s);
		}

		void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();

			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();

			Davgpooling_impl(CDATA(gy), gy_size, gx_size, n, gx);
		}

		void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);


			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();

			Fsumpooling_impl(x, MDATA(y), n, r, s);
		}

		void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();

			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();

			Dsumpooling_impl(CDATA(gy), gy_size, gx_size, gx);
		}
		
		void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y, int* index) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();


			Fmaxpooling_impl(x, MDATA(y), n, r, s, index);
		}

		void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const int dim0 = gy.shape()[0];

			Dmaxpooling_impl(CDATA(gy), gx, index, dim0);
		}

		void FMinPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();


			Fminpooling_impl(x, MDATA(y), n, r, s, index);
		}

		void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const int dim0 = gy.shape()[0];

			Dminpooling_impl(CDATA(gy), gx, index, dim0);
		}

		void copy_tensor(const LDG::Tensor &src, LDG::Tensor& dst) {
			malloc(dst, src.shape());
			cudaMemcpyAsync(
					MDATA(dst),
				   	CDATA(src), 
					sizeof(dtype) * src.shape().size(), 
					cudaMemcpyDeviceToDevice, 0);
		}

		vector<dtype> to_vector(const LDG::Tensor& x) {
			const std::uint32_t size = x.shape().size();
			vector<dtype> ret(size);
			/*
			cudaMemcpy(
					ret.data(), 
					CDATA(x), 
					sizeof(dtype) * size, 
					cudaMemcpyDeviceToHost);
					*/
			
			cudaMemcpyAsync(
					ret.data(), 
					CDATA(x), 
					sizeof(dtype) * size, 
					cudaMemcpyDeviceToHost, 0);
			return ret;
		}

		void to_vector(const vector<LDG::PTensor>& vec_tensor, vector<vector<dtype>* >& vec_data) {
			int size = vec_tensor.size();
			assert(size == vec_data.size());

			for(int idx = 0; idx < size; idx++) {
				int dim = vec_tensor[idx]->shape().size();
				vec_data[idx]->resize(dim);
				cudaMemcpyAsync(
						vec_data[idx]->data(),
						CDATA(*vec_tensor[idx]),
						sizeof(dtype) * dim,
						cudaMemcpyDeviceToHost, streams[idx % STREAM_NUM]);
			}
		}

		vector<dtype> to_vector(const LDG::Tensor& x, const vector<int>& cols) {
			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int col_num = cols.size();
			int r_size = xdim0 * col_num;

			int mem_size = sizeof(int) * col_num;
			IndexPtr gpu_cols;
			init_index_ptr(gpu_cols, col_num);
			cudaMemcpyAsync(gpu_cols.get_ptr(), cols.data(), mem_size, cudaMemcpyHostToDevice, 0);

			LDG::Tensor r_tensor;
			malloc(r_tensor, Shape({xdim0, col_num}));

			get_cols_impl(CDATA(x), MDATA(r_tensor), xdim0, xdim1, r_size, gpu_cols.get_ptr(), cols.size());

			vector<dtype> ret = to_vector(r_tensor);

			return ret;
		}


		void Fdropout(const vector<LDG::PTensor>& vec_x, dtype drop_rate, vector<LDG::PTensor>& vec_r) {
			int dim = vec_x[0]->shape()[0];
			int count = vec_x.size();
			for(int idx = 0; idx < count; idx++)
				assert(vec_x[idx]->shape()[0] == vec_r[idx]->shape()[0]);
			assert(count == vec_r.size());

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();
			
			ArrayPtr r_ptr(vec_r);
			dtype **r = (dtype**)r_ptr.get_ptr();

			Fdropout_impl(x, drop_rate, dim, count, r);		
		}

		void Fdropout(const vector<LDG::PTensor>& vec_x, dtype drop_rate, IndexPtr& mask, vector<LDG::PTensor>& vec_r) {
			DtypePtr mask_val;
			int dim = vec_x[0]->shape()[0];
			int count = vec_x.size();
			init_index_ptr(mask, count * dim);
			init_dtype_ptr(mask_val, count * dim);
			for(int idx = 0; idx < count; idx++)
				assert(vec_x[idx]->shape()[0] == vec_r[idx]->shape()[0]);

			assert(count == vec_r.size());
			assert(count * dim == mask_val.size);
			assert(count * dim == mask.size);

#if USE_FLOAT
			curandGenerateUniform(rndGen, mask_val.get_ptr(), mask_val.size);
#else
			curandGenerateUniformDouble(rndGen, mask_val.get_ptr(), mask_val.size);
#endif
			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();
			
			ArrayPtr r_ptr(vec_r);
			dtype **r = (dtype**)r_ptr.get_ptr();

			Fdropout_impl(x, drop_rate, mask_val.get_ptr(), mask.get_ptr(), dim, count, r);
		}

		void Ddropout(vector<LDG::PTensor>& vec_in_loss, IndexPtr& mask, vector<LDG::PTensor>& vec_loss) {
			int dim = vec_in_loss[0]->shape()[0];
			int count = vec_in_loss.size();

			assert(count == vec_loss.size());

			ArrayPtr in_loss_ptr(vec_in_loss);
			dtype **in_loss = (dtype**)in_loss_ptr.get_ptr();
			
			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			Ddropout_impl(in_loss, mask.get_ptr(), dim, count, loss);
		}

		void Ddropout(vector<LDG::PTensor>& vec_loss, IndexPtr& mask) {
			int dim = vec_loss[0]->shape()[0];
			int count = vec_loss.size();

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			Ddropout_impl(loss, mask.get_ptr(), dim, count);
		}
};

#endif
