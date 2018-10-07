#ifndef PRIMITIV_SHAPE_H_
#define PRIMITIV_SHAPE_H_

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <sstream>
#include "error.h"


/**
* Data structure to represent the shape of the node.
*
* Examples:
*   Shape()         == Shape({1, 1, 1, ...}, 1): scalar
*   Shape({})       == Shape({1, 1, 1, ...}, 1): scalar
*   Shape({n})      == Shape({n, 1, 1, ...}, 1): column vector
*   Shape({n, m})   == Shape({n, m, 1, ...}, 1): matrix
*   Shape({...}, k): k-parallelized data (mini-batch)
*/
class Shape {
public:
	static const int MAX_DEPTH = 8;

	Shape(const Shape &) = default;
	Shape(Shape &&) = default;
	Shape &operator=(const Shape &) = default;
	Shape &operator=(Shape &&);

	/**
	* Creates a new scalar Shape object.
	*/
	Shape();

	/**
	* Creates a new Shape object.
	* @param dims List of the dimension sizes.
	* @param batch Batch size.
	*/
	Shape(std::initializer_list<int> dims, int batch = 1);

	/**
	* Creates a new Shape object.
	* @param dims List of the dimension sizes.
	* @param batch Batch size.
	*/
	Shape(const std::vector<int> &dims, int batch = 1);

	/**
	* Returns the size of the i-th dimension.
	* @param i Dimension number to check.
	* @return Size of the i-th dimension.
	*/
	int operator[](int i) const;

	/**
	* Returns the dimension array.
	* @return Copy of the dimension array.
	*/
	const std::vector<int> dims() const;
	/**
	* Returns the depth (length of non-1 dimensions) of the shape.
	* @return The depth of the shape.
	*/
	int depth() const;

	/**
	* Returns the batch size.
	* @return Batch size.
	*/
	int batch() const;

	/**
	* Returns the number of elements in each sample.
	* This value is equal to the product of all dimensions.
	* @return Number of elements.
	*/
	int volume() const;

	/**
	* Returns the number of elements in 1 to specified dim.
	* @param dim Upper bound of the dimension.
	* @return `dims[0] * dims[1] * ... * dims[dim-1]`
	*/
	int lower_volume(int dim) const;

	/**
	* Returns the number of elements in all samples of the mini-batch.
	* This value is equal to `batch() * volume()`.
	* @return Number of elements.
	*/
	int size() const;

	/**
	* Returns a string representation of the shape.
	* The format is: "[n,m,...]xk"
	* @return Encoded string.
	*/
	std::string to_string() const;

	/**
	* Compares this and other shape.
	* @param rhs Shape object to compare.
	* @return true if this and rhs are same, false otherwise.
	*/
	bool operator==(const Shape &rhs) const;

	/**
	* Compares this and other shape.
	* @param rhs Shape object to compare.
	* @return true if this and rhs are not same, false otherwise.
	*/
	bool operator!=(const Shape &rhs) const;

	/**
	* Checks whether the shape has minibatch or not.
	* @return true if the shape has minibatch, false otherwise.
	*/
	bool has_batch() const;

	/**
	* Checks whether two batch size is compatible (broadcastable) or not.
	* @param rhs Shape object to compare.
	* @return true if both batch size is compatible, false otherwise.
	*/
	bool has_compatible_batch(const Shape &rhs) const;

	/**
	* Checks whether the shape is a scalar or not.
	* @return true if the shape is a scalar, false otherwise.
	*/
	bool is_scalar() const;

	/**
	* Checks whether the shape is a column vector or not.
	* @return true if the shape is a column vector, false otherwise.
	*/
	bool is_column_vector() const;

	/**
	* Checks whether the shape is a vector or a matrix, or not.
	* @return true if the shape is a vector or a matrix, false otherwise.
	*/
	bool is_matrix() const;

	/**
	* Checks whether two shapes have completely same dimensions.
	* @param rhs Shape object to compare.
	* @return true if both shape have same dimensions, false otherwise.
	*/
	bool has_same_dims(const Shape &rhs) const;

	/**
	* Checks whether two shapes have same dimensions without an axis.
	* (LOO: leave one out)
	* @param rhs Shape object to compare.
	* @param dim Dimension to be ignored.
	* @return true if both shape have same dimensions regardless the dimension
	*         `dim`, false otherwise.
	*/
	bool has_same_loo_dims(const Shape &rhs, int dim) const;

	/**
	* Creates a new shape which have one different dimension.
	* @param dim Dimension to be changed.
	* @param m New size of the dimension `dim`.
	* @return New shape.
	*/
	Shape resize_dim(int dim, int m) const;

	/**
	* Creates a new shape which have specified batch size.
	* @param batch New batch size.
	* @return New shape.
	*/
	Shape resize_batch(int batch) const;

	/**
	* Directly updates a specified dimension.
	* @param dim Dimension to be updated.
	* @param m New size of the dimension `dim`.
	*/
	void update_dim(int dim, int m);

	/**
	* Directly updates the batch size.
	* @param batch New batch size.
	*/
	void update_batch(int batch);

private:
	std::array<int, MAX_DEPTH> dims_;
	int depth_;
	int batch_;
	int volume_;
};

Shape::Shape(std::initializer_list<int> dims, int batch)
	: depth_(0), batch_(batch), volume_(1) {
	if (dims.size() > MAX_DEPTH) {
		PRIMITIV_THROW_ERROR(
			"Exceeds dimension depth limit at Shape::Shape()."
			" depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
	}
	for (const int d : dims) {
		dims_[depth_++] = d;
		volume_ *= d;
	}
	//while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
	if (volume_ == 0 || batch_ == 0) {
		PRIMITIV_THROW_ERROR("Invalid shape: " << to_string());
	}
}

Shape::Shape(const std::vector<int> &dims, int batch)
	: depth_(0), batch_(batch), volume_(1) {
	if (dims.size() > MAX_DEPTH) {
		PRIMITIV_THROW_ERROR(
			"Exceeds dimension depth limit at Shape::Shape()."
			" depth: " << depth_ << " > MAX_DEPTH: " << MAX_DEPTH);
	}
	for (const int d : dims) {
		dims_[depth_++] = d;
		volume_ *= d;
	}
	//while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
	if (volume_ == 0 || batch_ == 0) {
		PRIMITIV_THROW_ERROR("Invalid shape: " << to_string());
	}
}

Shape &Shape::operator=(Shape &&src) {
	if (&src != this) {
		dims_ = std::move(src.dims_);
		depth_ = src.depth_;
		batch_ = src.batch_;
		volume_ = src.volume_;
	}
	return *this;
}

std::string Shape::to_string() const {
	std::stringstream s;
	s << '[';
	for (int i = 0; i < depth_; ++i) {
		if (i > 0) {
			s << ',';
		}
		s << dims_[i];
	}
	s << "]x" << batch_;
	return s.str();
}

bool Shape::has_same_loo_dims(const Shape &rhs, int dim) const {
	int nl = depth_ == dim + 1 ? dim : depth_;
	while (nl > 0 && dims_[nl - 1] == 1) --nl;
	int nr = rhs.depth_ == dim + 1 ? dim : rhs.depth_;
	while (nr > 0 && rhs.dims_[nr - 1] == 1) --nr;
	bool p = nl == nr;
	for (int i = 0; i < nl; ++i) {
		p = p && (dims_[i] == rhs.dims_[i] || i == dim);
	}
	return p;
}

Shape Shape::resize_dim(int dim, int m) const {
	Shape ret = *this;
	ret.update_dim(dim, m);
	return ret;
}

Shape Shape::resize_batch(int batch) const {
	Shape ret = *this;
	ret.update_batch(batch);
	return ret;
}

void Shape::update_dim(int dim, int m) {
	if (dim >= MAX_DEPTH) {
		PRIMITIV_THROW_ERROR(
			"Exceeds dimension depth limit at Shape::update_dim()."
			" dim: " << dim << " >= MAX_DEPTH: " << MAX_DEPTH);
	}
	if (m == 0) PRIMITIV_THROW_ERROR("Could not set each dimension to 0.");
	if (dim >= depth_) {
		int new_depth = dim + 1;
		for (int i = depth_; i < new_depth; ++i) dims_[i] = 1;
		depth_ = new_depth;
	}
	volume_ = (volume_ / dims_[dim]) * m;
	dims_[dim] = m;
	while (depth_ > 0 && dims_[depth_ - 1] == 1) --depth_;
}

void Shape::update_batch(int batch) {
	if (batch == 0) PRIMITIV_THROW_ERROR("Could not set the batch size to 0.");
	batch_ = batch;
}


Shape::Shape() : depth_(0), batch_(1), volume_(1) {}

int Shape::operator[](int i) const { return i < depth_ ? dims_[i] : 1; }

const std::vector<int> Shape::dims() const {
	return std::vector<int>(&dims_[0], &dims_[depth_]);
}

int Shape::depth() const { return depth_; }

int Shape::batch() const { return batch_; }

int Shape::volume() const { return volume_; }

int Shape::lower_volume(int dim) const {
	int ret = 1, lim = min(dim, depth_);
	for (int i = 0; i < lim; ++i) ret *= dims_[i];
		return ret;
}

int Shape::size() const { return batch_ * volume_; }

bool Shape::operator==(const Shape &rhs) const {
	return has_same_dims(rhs) && batch_ == rhs.batch_;
}

bool Shape::operator!=(const Shape &rhs) const { return !operator==(rhs); }

bool Shape::has_batch() const { return batch_ > 1; }

bool Shape::has_compatible_batch(const Shape &rhs) const {
	return batch_ == rhs.batch_ || batch_ == 1 || rhs.batch_ == 1;
}

bool Shape::is_scalar() const { return depth() == 0; }

bool Shape::is_column_vector() const { return depth() <= 1; }

bool Shape::is_matrix() const { return depth() <= 2; }

bool Shape::has_same_dims(const Shape &rhs) const {
	bool ok = true;
	for (int i = 0; i < depth_; ++i) ok = ok && dims_[i] == rhs.dims_[i];
	return ok && depth_ == rhs.depth_;
}
#endif 

