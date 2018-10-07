#ifndef EIGEN_DEVICE
#define EIGEN_DEVICE

#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include "memory_pool.h"
#include "Device.h"
#include "MyLib.h"
#include "functions.h"

using namespace Eigen;

typedef Eigen::TensorMap<Eigen::Tensor<dtype, 1>> Vec;
typedef Eigen::Map<Matrix<dtype, Dynamic, Dynamic, ColMajor>> Mat;

static void *my_alloc(std::size_t size) {
	void *ptr;
	ptr = malloc(size);
	return ptr;
}

static void my_delete(void *ptr) {
	free(ptr);
}

MemoryPool* get_mem_pool() {
	static MemoryPool mem_pool(my_alloc, my_delete);
	return &mem_pool;
}

#define MEM_POOL get_mem_pool()


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
			memcpy(v_data, (dtype**)vec_ptr.data(), mem_size);
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
				memcpy(v_data, (dtype**)vec_ptr.data(), mem_piece_size);

				vec_vec_ptr[idx] = v_data;
			}
			memcpy(v_v_data, (dtype***)vec_vec_ptr.data(), mem_size);
		}

		void *get_ptr() {
			return ptr_.get();
		}
};

class EigenDevice :public Device {
	private:


		Mat mat(LDG::Tensor& x) {
			return Mat(MDATA(x), x.shape()[0], x.shape()[1]);
		}

		Vec vec(LDG::Tensor& x) {
			return Vec(MDATA(x), x.shape().size());
		}

		const Mat mat(const LDG::Tensor& x) const {
			return Mat((dtype*)x.handle_.get(), x.shape()[0], x.shape()[1]);
		}

		const Vec vec(const LDG::Tensor& x) const {
			return Vec((dtype*)x.handle_.get(), x.shape().size());
		}

		std::default_random_engine generator;
		std::uniform_real_distribution<dtype>* distribution;
	public:

		EigenDevice() {
			distribution = new uniform_real_distribution<dtype>(0.0, 1.0);
		}

		~EigenDevice() {
			delete distribution;
		}

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

		void malloc(LDG::Tensor &t, const Shape &shape){
			t.device_type = EIGEN;
			t.shape_ = shape;
			int size = shape.size();
			int mem_size = sizeof(dtype) * size;

			t.handle_ = MEM_POOL->allocate(mem_size);
		}

		void set(LDG::Tensor &t, dtype val) {
			vec(t).setConstant(val);
		}

		void set(vector<LDG::PTensor> &vec_t, dtype val){
			int size = vec_t.size();
			for(int idx = 0; idx < size; idx++) {
				set(*vec_t[idx], val);
			}
		}

		void set(LDG::Tensor &t, const vector<dtype>& vec_val){
			int size = t.shape().size();
			assert(vec_val.size() == size);
			memcpy(MDATA(t), vec_val.data(), sizeof(dtype) * vec_val.size());
		}

		void set(LDG::Tensor &t, const dtype* host_data, int h_size){
			int size = t.shape().size();
			assert (size == h_size);
			memcpy(MDATA(t), host_data, sizeof(dtype) * h_size);
		}

		void zero(LDG::Tensor &t){
			set(t, 0.0);
		}

		void set_col(LDG::Tensor &t, int col, dtype val){
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];
			assert(col < dim1);
			mat(t).col(col).setConstant(val);
		}

		void set_cols(LDG::Tensor &t, const vector<int>& cols, const vector<dtype>& vals){
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];

			int col_num = cols.size();
			assert(dim0 * col_num == vals.size());
			for(int idx = 0; idx < col_num; idx++)
				assert(cols[idx] < dim1);

			Mat mat_t = mat(t);
			for(int idx = 0; idx < col_num; idx++) {
				int c = cols[idx];
				for(int idy = 0; idy < dim0; idy++) {
					mat_t.col(c)[idy] = vals[idx * dim0 + idy];
				}
			}	
		}
		// void copy_data(const LDG::Tensor& src, LDG::Tensor& dst){}

		void copy_tensor(const LDG::Tensor& src, LDG::Tensor& dst){
			malloc(dst, src.shape());
			memcpy(
					MDATA(dst),
					CDATA(src), 
					sizeof(dtype) * src.shape().size());
		} //

		vector<dtype> to_vector(const LDG::Tensor& x){
			const std::uint32_t size = x.shape().size();
			vector<dtype> ret(size);
			memcpy(
					ret.data(), 
					CDATA(x), 
					sizeof(dtype) * size);
			return ret;
		}

		void to_vector(const vector<LDG::PTensor>& vec_tensor, vector<vector<dtype>* >& vec_data){
			int size = vec_tensor.size();
			assert(size == vec_data.size());

			for(int idx = 0; idx < size; idx++) {
				int dim = vec_tensor[idx]->shape().size();
				vec_data[idx]->resize(dim);
				memcpy(
						vec_data[idx]->data(),
						CDATA(*vec_tensor[idx]),
						sizeof(dtype) * dim
					  );
			}
		}

		vector<dtype> to_vector(const LDG::Tensor& x, const vector<int>& cols) {
			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int col_num = cols.size();

			const Mat& mat_x = mat(x);

			LDG::Tensor r_tensor;
			malloc(r_tensor, Shape({xdim0, col_num}));
			Mat mat_r = mat(r_tensor);
			for(int idx = 0; idx < col_num; idx++) {
				int c = cols[idx];
				assert(c < xdim1);
				mat_r.col(idx) = mat_x.col(c);
			}
			vector<dtype> ret = to_vector(r_tensor);
			return ret;
		}

		void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r){
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			malloc(r, Shape({dim0, 1}));
			assert(col < dim1);
			mat(r) = mat(x).col(col);
		}

		void get_cols(const LDG::Tensor& x, int* cols, int col_num, LDG::Tensor& r){
			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];

			const Mat& mat_x = mat(x);

			malloc(r, Shape({xdim0, col_num}));
			Mat mat_r = mat(r);
			for(int idx = 0; idx < col_num; idx++) {
				int c = cols[idx];
				assert(c < xdim1);
				mat_r.col(idx) = mat_x.col(c);
			}
		}

		void FLookup(const LDG::Tensor& x, const int* cols, int col_num, vector<LDG::PTensor>& vec_r) {
			assert(vec_r.size() == col_num);
			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];

			const Mat& mat_x = mat(x);
			for(int idx = 0; idx < col_num; idx++) {
				int c = cols[idx];
				assert(c < xdim1);
				mat(*vec_r[idx]) = mat_x.col(c);
			}
		}

		void DLookup(LDG::Tensor& gx, const int* cols, int col_num, vector<LDG::PTensor>& vec_loss) {
			assert(vec_loss.size() == col_num);
			int gxdim0 = gx.shape()[0];
			int gxdim1 = gx.shape()[1];

			Mat mat_gx = mat(gx);
			for(int idx = 0; idx < col_num; idx++) {
				int c = cols[idx];
				assert(c < gxdim1);
				mat_gx.col(c) += mat(*vec_loss[idx]);
			}
		}

		void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_r){
			int size = vec_cols.size();
			FLookup(x, vec_cols.data(), size, vec_r);
		}

		void DLookup(LDG::Tensor& gx, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_loss){
			int size = vec_cols.size();
			DLookup(gx, vec_cols.data(), size, vec_loss);
		}

		void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper){
			int size = shape.size();
			int memsize = sizeof(dtype) * size;
			vector<dtype> host_data(size);
			dtype min = lower, max = upper;
			for (int i = 0; i < size; i++) {
				host_data[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
			}

			malloc(t, shape);
			memcpy(MDATA(t), host_data.data(), memsize);
		}

		void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p){}
		void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}
		void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}

		void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Ftanh(*vec_x[idx], *vec_r[idx]);
		}

		void Dtanh(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r){
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Dtanh(*vec_x[idx], *vec_y[idx], *vec_r[idx]);
		}

		void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Fsigmoid(*vec_x[idx], *vec_r[idx]);
		}
		void Dsigmoid(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r){
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Dsigmoid(*vec_x[idx], *vec_y[idx], *vec_r[idx]);
		}

		void Fequal(const LDG::Tensor& x, LDG::Tensor& r){}

		void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x).unaryExpr(ptr_fun(ftanh));
		}

		void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x).unaryExpr(ptr_fun(fsigmoid));
		}

		void Frelu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fexp(const LDG::Tensor& x, LDG::Tensor& r){}
		void Flog(const LDG::Tensor& x, LDG::Tensor& r){}

		void Fsquare(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x).square();
		}

		void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x).sqrt();
		}

		void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			malloc(r, x.shape());
			assert(x_size == y_size);
			vec(r) = vec(x).binaryExpr(vec(y), ptr_fun(dtanh));
		}

		void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			assert(x_size == y_size);
			malloc(r, x.shape());
			vec(r) = vec(x).binaryExpr(vec(y), ptr_fun(dsigmoid));
		}
		void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_dim = x.shape()[0];
			int y_dim = y.shape()[0];
			int r_dim = r.shape()[0];
			malloc(r, x.shape());
			assert(x_dim == y_dim);
			vec(r) = vec(x) + vec(y); 
		}

		// LDG::Tensor Fadd(const LDG::Tensor& x, const LDG::Tensor& y){}

		void Fadd(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r){
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Fadd(*vec_x[idx], *vec_y[idx], *vec_r[idx]);
		}

		// void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r){}

		void Fadd_inplace(LDG::Tensor& x, const LDG::Tensor& y){
			int x_dim = x.shape()[0];
			int y_dim = y.shape()[0];

			assert(x_dim == y_dim);
			vec(x) += vec(y);
		}

		void Fadd_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y){
			int size  = vec_x.size();
			assert(size == vec_y.size());
			for(int idx = 0; idx < size; idx++)
				Fadd_inplace(*vec_x[idx], *vec_y[idx]);
		}

		void Dadd_inplace(LDG::Tensor& in_loss, const LDG::Tensor& loss) {
			const int dim = loss.shape()[0];
			assert(dim == loss.shape()[0]);
			const int n = loss.shape()[1];

			Mat mat_in_loss = mat(in_loss);
			Mat mat_loss = mat(loss);
			for(int idx = 0; idx < n; idx++) {
				mat_in_loss += mat_loss.col(idx);
			}

		}

		void Dadd_inplace(vector<LDG::PTensor>& vec_in_loss, const LDG::Tensor& loss){
			Mat mat_loss = mat(loss);
			int size = vec_in_loss.size();
			for(int idx = 0; idx < size; idx++) {
				mat(*vec_in_loss[idx]) += mat_loss.col(idx);
			}
		}

		void Fadd(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_y){
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

			for(int idx = 0; idx < count; idx++) {
				Mat mat_y = mat(*vec_y[idx]);
				mat_y = mat(*vec_vec_x[idx][0]);
				for(int idy = 1; idy < n; idy++) {
					mat_y += mat(*vec_vec_x[idx][idy]);
				}
			}
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
			for(int idx = 0; idx < count; idx++) {
				Mat mat_loss = mat(*vec_loss[idx]);
				for(int idy = 0; idy < n; idy++) {
					mat(*vec_vec_in_loss[idx][idy]) += mat_loss;
				}
			}


		}

		void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			assert(x_size == y_size) ;
			vec(r) = vec(x) - vec(y);
		}

		void Fsubtract_inplace(LDG::Tensor& x, const LDG::Tensor& y){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			assert(x_size == y_size) ;
			vec(x) -= vec(y);
		}


		void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			assert(x_size == y_size) ;
			vec(r) = vec(x) / vec(y);
		}

		// void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false){}

		void Dmatmul(const LDG::Tensor& x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false){
			LDG::Tensor y;
			int y_size = vec_y.size();
			int y_dim = vec_y[0]->shape()[0];
			concat(vec_y, y);

			Dmatmul(x, y, r, tx, ty);
		}

		// void Dmatmul(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false){}

		void Dmatmul(const LDG::Tensor& x, const LDG::Tensor &y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false){
			LDG::Tensor r;
			int r_size = vec_r.size();
			int r_dim = vec_r[0]->shape()[0];

			malloc(r, Shape({r_dim, r_size}));
			Fmatmul(x, y, r, tx, ty);
			Dadd_inplace(vec_r, r);
		}

		void Fdot(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false){
			int n = vec_x.size();
			assert(vec_y.size() == n && vec_r.size() == n);
			for(int idx = 0; idx < n; idx++)
				Fmatmul(*vec_x[idx], *vec_y[idx], *vec_r[idx], tx, ty);
		}

		void Ddot(vector<LDG::PTensor> &vec_in_loss, const vector<LDG::PTensor> &vec_val, const vector<LDG::PTensor> &vec_loss){
			int n = vec_loss.size();
			assert(vec_val.size() == n && vec_loss.size() == n);
			for(int idx = 0; idx < n; idx++)
				vec(*vec_in_loss[idx]) += vec(*vec_loss[idx]).data()[0] * vec(*vec_loss[idx]);
		}

		void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r, bool tx = false, bool ty = false){
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = ty ? y.shape()[0] : y.shape()[1];

			malloc(r, Shape({m, n}));

			if(tx == true && ty == false)
				mat(r) = mat(x).transpose() * mat(y);
			if(tx == false && ty == true)
				mat(r) = mat(x) * mat(y).transpose();
			if(tx == true && ty == true)
				mat(r) = mat(x).transpose() * mat(y).transpose();
			if(tx == false && ty == false)
				mat(r) = mat(x) * mat(y);
		}

		void Dmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r, bool tx = false, bool ty = false){
			LDG::Tensor tmp_r;
			Fmatmul(x, y, tmp_r, tx, ty);
			vec(r) += vec(tmp_r);
		}

		void Dmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false){
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

		void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			assert(x_size == y_size);
			malloc(r, x.shape());
			vec(r) = vec(x) * vec(y);
		}

		void Fmultiply(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r){
			int size  = vec_x.size();
			assert(size == vec_r.size());
			for(int idx = 0; idx < size; idx++)
				Fmultiply(*vec_x[idx], *vec_y[idx], *vec_r[idx]);
		}

		void Fmultiply_inplace(LDG::Tensor& x, const LDG::Tensor& y){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			assert(x_size == y_size);
			vec(x) *= vec(y);
		}

		void Fmultiply_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y){
			int size  = vec_x.size();
			assert(size == vec_y.size());
			for(int idx = 0; idx < size; idx++)
				Fmultiply_inplace(*vec_x[idx], *vec_y[idx]);
		}

		void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x) + y;
		}

		void Fadd_scalar_inplace(LDG::Tensor& x, const dtype y){
			int x_size = x.shape().size();
			vec(x) = vec(x) + y;
		}

		void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			vec(r) = vec(x) * y;
		}

		void Fmultiply_scalar(const LDG::Tensor& x, const LDG::Tensor& scalar, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			assert(scalar.shape().size() == 1);
			vec(r) = vec(x) * vec(scalar).data()[0];
		}

		void Fmultiply_scalar_inplace(LDG::Tensor& x, const dtype y){
			int x_size = x.shape().size();
			vec(x) = vec(x) * y;
		}

		void Fadd_col(LDG::Tensor& x, const LDG::Tensor& y_col, int col){
			auto x_dims = x.shape().dims();
			int dim0 = x_dims[0];
			int dim1 = x_dims[1];
			assert(dim0 == y_col.shape()[0]);
			assert(y_col.shape()[1] == 1);
			assert(col < dim1);
			mat(x).col(col) += mat(y_col);
		}

		void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){
			const Shape xshape = vec_x[0]->shape();
			int src_dim = xshape[0];
			int n = vec_x.size();

			malloc(r, Shape({src_dim, n}));

			Mat mat_r = mat(r);
			for(int idx = 0; idx < n; idx++) {
				mat_r.col(idx) = mat(*vec_x[idx]); 
			}
		}

		void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
			int n = vec_x.size();
			const Mat& mat_r = mat(r);
			for(int idx = 0; idx < n; idx++) {
				mat(*vec_x[idx]) = mat_r.col(idx);
			}
		}

		void Fconcat(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_r){
			int count = vec_vec_x.size();
			int n = vec_vec_x[0].size();
			assert(count == vec_r.size());

			ArrayPtr src_ptr(vec_vec_x);
			dtype ***src = (dtype ***)src_ptr.get_ptr();

			ArrayPtr dst_ptr(vec_r);
			dtype **dst = (dtype **)dst_ptr.get_ptr();

			vector<int> dims(n);
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_vec_x[0][idx]->shape()[0];
				//cout << vec_vec_x[0][idx]->shape().to_string() << endl;
				dims[idx] = dim;
			}

			for(int idx = 0; idx < count; idx++) {
				int offset = 0;
				for(int idy = 0; idy < n; idy++) {
					int dim = dims[idy];
					for(int idz = 0; idz < dim; idz++) {
						dst[idx][offset + idz] = src[idx][idy][idz];
					}
					offset += dim;
				}
			}
		}

		void Dconcat(vector<vector<LDG::PTensor> >& vec_vec_in_loss, const vector<LDG::PTensor>& vec_loss) {
			int n = vec_vec_in_loss[0].size();
			int count = vec_loss.size();
			assert(count == vec_vec_in_loss.size());

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype **)loss_ptr.get_ptr();

			ArrayPtr in_loss_ptr(vec_vec_in_loss);
			dtype ***in_loss = (dtype ***)in_loss_ptr.get_ptr();
			vector<int> dims(n);
			for(int idx = 0; idx < n; idx++) {
				int dim = vec_vec_in_loss[0][idx]->shape()[0];
				//cout << vec_vec_x[0][idx]->shape().to_string() << endl;
				dims[idx] = dim;
			}

			for(int idx = 0; idx < count; idx++) {
				int offset = 0;
				for(int idy = 0; idy < n; idy++) {
					int dim = dims[idy];
					for(int idz = 0; idz < dim; idz++) {
						in_loss[idx][idy][idz] += loss[idx][offset + idz];
					}
					offset += dim;
				}
			}

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

			randGenerationUniform(*distribution, generator, mask_val.get_ptr(), mask_val.size);
			dtype* mask_val_ptr = mask_val.get_ptr();
			int* mask_ptr = mask.get_ptr();
			//for(int idx = 0; idx < mask_val.size; idx++)
				//cout << v[idx] << " ";
			//cout << endl;

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();
			
			ArrayPtr r_ptr(vec_r);
			dtype **r = (dtype**)r_ptr.get_ptr();
			for(int idx = 0; idx < count; idx++) {
				for(int idy = 0; idy < dim; idy++) {
					int index = idx * dim + idy;
					mask_ptr[index] = mask_val_ptr[index] > drop_rate ? 1 : 0;
					r[idx][idy] = x[idx][idy] * mask_ptr[index];
				}
			}
		}

		void Fdropout(const vector<LDG::PTensor>& vec_x, dtype drop_rate, vector<LDG::PTensor>& vec_r) {
			int dim = vec_x[0]->shape()[0];
			int count = vec_x.size();
			//for(int idx = 0; idx < mask_val.size; idx++)
				//cout << v[idx] << " ";
			//cout << endl;

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype**)x_ptr.get_ptr();

			ArrayPtr r_ptr(vec_r);
			dtype **r = (dtype**)r_ptr.get_ptr();

			dtype rate = 1 - drop_rate;
			for(int idx = 0; idx < count; idx++) {
				for(int idy = 0; idy < dim; idy++) {
					r[idx][idy] = x[idx][idy] * rate;
				}
			}
		}

		void Ddropout(vector<LDG::PTensor>& vec_in_loss, IndexPtr& mask, vector<LDG::PTensor>& vec_loss) {
			int dim = vec_in_loss[0]->shape()[0];
			int count = vec_in_loss.size();

			assert(count == vec_loss.size());

			ArrayPtr in_loss_ptr(vec_in_loss);
			dtype **in_loss = (dtype**)in_loss_ptr.get_ptr();
			
			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			int* mask_ptr = mask.get_ptr();

			for(int idx = 0; idx < count; idx++) {
				for(int idy = 0; idy < dim; idy++) {
					int index = idx * dim + idy;
					in_loss[idx][idy] += loss[idx][idy] * mask_ptr[index];
				}
			}
		}

		void Ddropout(vector<LDG::PTensor>& vec_loss, IndexPtr& mask) {
			int dim = vec_loss[0]->shape()[0];
			int count = vec_loss.size();

			assert(count == vec_loss.size());

			ArrayPtr loss_ptr(vec_loss);
			dtype **loss = (dtype**)loss_ptr.get_ptr();

			int* mask_ptr = mask.get_ptr();
			
			for(int idx = 0; idx < count; idx++) {
				for(int idy = 0; idy < dim; idy++) {
					int index = idx * dim + idy;
					loss[idx][idy] *= mask_ptr[index];
				}
			}
		}

		void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			int n = vec_x.size();
			Vec v_y = vec(y);
			v_y = vec(*vec_x[0]);
			for(int idx = 1; idx < n; idx++) {
				v_y += vec(*vec_x[idx]);
			}

			v_y = v_y / (dtype)n;
		}

		void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			int n = vec_gx.size();
			Vec v_gy = vec(gy);
			for(int idx = 0; idx < n; idx++) {
				vec(*vec_gx[idx]) += v_gy / (dtype) n;
			}
		}

		void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			int n = vec_x.size();
			Vec v_y = vec(y);
			v_y = vec(*vec_x[0]);
			for(int idx = 1; idx < n; idx++) {
				v_y += vec(*vec_x[idx]);
			}
		}

		void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			int n = vec_gx.size();
			Vec v_gy = vec(gy);
			for(int idx = 0; idx < n; idx++) {
				vec(*vec_gx[idx]) += v_gy;
			}
		}

		void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());
			const int r = y.shape().size();

			dtype* y_ptr = MDATA(y);

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();
			for(int idy = 0; idy < r; idy++) {
				dtype max = x[0][idy];
				index[idy] = 0;
				for(int idx = 1; idx < n; idx++) {
					if(x[idx][idy] > max) {
						max = x[idx][idy];
						index[idy] = idx;
					}
				}
				y_ptr[idy] = max;
			}
		}

		void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const dtype* gy_ptr = CDATA(gy);
			const int r = gy.shape().size();

			for(int idx = 0; idx < r; idx++) {
				gx[index[idx]][idx] += gy_ptr[idx];
			}
		}

		void FMinPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) { 
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());
			const int r = y.shape().size();

			dtype* y_ptr = MDATA(y);

			ArrayPtr x_ptr(vec_x);
			dtype **x = (dtype **)x_ptr.get_ptr();
			for(int idy = 0; idy < r; idy++) {
				dtype min = x[0][idy];
				index[idy] = 0;
				for(int idx = 1; idx < n; idx++) {
					if(x[idx][idy] < min) {
						min = x[idx][idy];
						index[idy] = idx;
					}
				}
				y_ptr[idy] = min;
			}
		}

		void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			ArrayPtr gx_ptr(vec_gx);
			dtype **gx = (dtype **)gx_ptr.get_ptr();

			const dtype* gy_ptr = CDATA(gy);
			const int r = gy.shape().size();

			for(int idx = 0; idx < r; idx++) {
				gx[index[idx]][idx] += gy_ptr[idx];
			}
		}

};

#endif
