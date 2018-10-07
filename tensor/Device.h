#ifndef LDG_DEVICE
#define LDG_DEVICE

#include "LDG_Tensor.h"
#include "MyLib.h"


class Device {
public:
	int device_type;
public:
	virtual void init_index_ptr(IndexPtr& index_ptr, int size) = 0;
	virtual void init_dtype_ptr(DtypePtr& dtype_ptr, int size) = 0;
	virtual void init(LDG::Tensor &t, const Shape &shape) = 0;
	virtual void malloc(LDG::Tensor &t, const Shape &shape) = 0;
	virtual void set(LDG::Tensor &t, dtype val) = 0;
	virtual void set(vector<LDG::PTensor> &vec_t, dtype val) = 0;

	virtual void set(LDG::Tensor &t, const vector<dtype>& vec_val) = 0;
	virtual void set(LDG::Tensor &t, const dtype* host_data, int h_size) = 0;

	virtual void zero(LDG::Tensor &t) = 0;


	virtual void set_col(LDG::Tensor &t, int col, dtype val) = 0;
	virtual void set_cols(LDG::Tensor &t, const vector<int>& cols, const vector<dtype>& vals) = 0;
	//virtual void copy_data(const LDG::Tensor& src, LDG::Tensor& dst) = 0;

	virtual void copy_tensor(const LDG::Tensor& src, LDG::Tensor& dst) = 0; //

	virtual vector<dtype> to_vector(const LDG::Tensor& x) = 0;

	virtual void to_vector(const vector<LDG::PTensor>& vec_tensor, vector<vector<dtype>* >& vec_data) = 0;

	virtual vector<dtype> to_vector(const LDG::Tensor& x, const vector<int>& cols) = 0;

	virtual void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) = 0;
	virtual void get_cols(const LDG::Tensor& x, int* cols, int col_num, LDG::Tensor& r) = 0;
	virtual void FLookup(const LDG::Tensor& x, const int* cols, int col_num, vector<LDG::PTensor>& vec_r) = 0;
	virtual void DLookup(LDG::Tensor& gx, const int* cols, int col_num, vector<LDG::PTensor>& vec_loss) = 0;

	virtual void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_r) = 0;
	virtual void DLookup(LDG::Tensor& gx, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_loss) = 0;

	virtual void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) = 0;
	virtual void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p) = 0;
	virtual void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) = 0;
	virtual void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) = 0;

	virtual void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) = 0;
	virtual void Dtanh(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) = 0;

	virtual void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) = 0;
	virtual void Dsigmoid(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) = 0;

	virtual void Fequal(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Frelu(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fexp(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Flog(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	
	virtual void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) = 0;

	virtual void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	virtual void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	//virtual LDG::Tensor Fadd(const LDG::Tensor& x, const LDG::Tensor& y) = 0;

	virtual void Fadd(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) = 0;

	//virtual void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r) = 0;

	virtual void Fadd_inplace(LDG::Tensor& x, const LDG::Tensor& y) = 0;

	virtual void Fadd_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) = 0;

	virtual void Dadd_inplace(LDG::Tensor& in_loss, const LDG::Tensor& loss) = 0;

	virtual void Dadd_inplace(vector<LDG::PTensor>& vec_in_loss, const LDG::Tensor& loss) = 0;

	virtual void Fadd(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_y) = 0;

	virtual void Dadd(vector<vector<LDG::PTensor> >& vec_vec_in_loss, const vector<LDG::PTensor>& vec_loss) = 0;

	virtual void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	virtual void Fsubtract_inplace(LDG::Tensor& x, const LDG::Tensor& y) = 0;


	virtual void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	//virtual void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;

	virtual void Dmatmul(const LDG::Tensor& x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;

	//virtual void Dmatmul(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;

	virtual void Dmatmul(const LDG::Tensor& x, const LDG::Tensor &y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) = 0;
			
	virtual void Fdot(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) = 0;

	virtual void Ddot(vector<LDG::PTensor> &vec_in_loss, const vector<LDG::PTensor> &vec_val, const vector<LDG::PTensor> &vec_loss) = 0;

	virtual void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;

	virtual void Dmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;

	virtual void Dmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) = 0;

	virtual void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	virtual void Fmultiply(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) = 0;

	virtual void Fmultiply_inplace(LDG::Tensor& x, const LDG::Tensor& y) = 0;

	virtual void Fmultiply_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) = 0;

	virtual void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) = 0;

	virtual void Fadd_scalar_inplace(LDG::Tensor& x, const dtype y) = 0;

	virtual void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) = 0;

	virtual void Fmultiply_scalar(const LDG::Tensor& x, const LDG::Tensor& scalar, LDG::Tensor& r) = 0;

	virtual void Fmultiply_scalar_inplace(LDG::Tensor& x, const dtype y) = 0;

	virtual void Fadd_col(LDG::Tensor& x, const LDG::Tensor& y_col, int col) = 0;

	//virtual void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	//virtual void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
		//LDG::Tensor& gx) = 0;

	virtual void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;

	virtual void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) = 0;

	virtual void Fconcat(const vector<vector<LDG::PTensor> >& vec_vec_x, vector<LDG::PTensor>& vec_r) = 0;
	
	virtual void Dconcat(vector<vector<LDG::PTensor> >& vec_vec_in_loss, const vector<LDG::PTensor>& vec_loss) = 0;

	//virtual void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fdropout(const vector<LDG::PTensor>& vec_x, dtype drop_rate, IndexPtr& mask, vector<LDG::PTensor>& vec_r) = 0;

	virtual void Fdropout(const vector<LDG::PTensor>& vec_x, dtype drop_rate, vector<LDG::PTensor>& vec_r) = 0;

	virtual void Ddropout(vector<LDG::PTensor>& vec_in_loss, IndexPtr& mask, vector<LDG::PTensor>& vec_loss) = 0;

	virtual void Ddropout(vector<LDG::PTensor>& vec_loss, IndexPtr& mask) = 0;

	virtual void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) = 0;

	virtual void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) = 0;

	virtual void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) = 0;

	virtual void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) = 0;

	virtual void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) = 0;

	virtual void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) = 0;

	virtual void FMinPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) = 0;

	virtual void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) = 0;

	void unaryExp(const LDG::Tensor& x, LDG::Tensor& r, 
			Device *dev, void (Device::*f)(const LDG::Tensor&, LDG::Tensor& )) {
		(dev->*f)(x, r);
	}

	void binaryExp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r, 
			Device *dev, void (Device::*f)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor& )) {
		(dev->*f)(x, y, r);
	}

	void binaryExp(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r, 
			Device *dev, void (Device::*f)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>& )) {
		(dev->*f)(vec_x, vec_y, vec_r);
	}

	void unaryExp(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r, 
			Device *dev, void (Device::*f)(const vector<LDG::PTensor>&, vector<LDG::PTensor>& )) {
		(dev->*f)(vec_x, vec_r);
	}


	const void* Handle(const LDG::Tensor& x){
		return x.get_handle();
	}

	void* Mutable_Handle(LDG::Tensor& x){
		if(x.handle_.use_count() > 1) {
			copy_tensor(x, x);
		}
		return x.handle_.get();
	}

#define CDATA(x) static_cast<const dtype* >(Handle(x))
#define MDATA(x) static_cast<dtype* >(Mutable_Handle(x))
};

#endif // ! Device
