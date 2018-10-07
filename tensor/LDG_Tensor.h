#ifndef LDG_TENSOR
#define LDG_TENSOR

#include <memory>

#include "MyLib.h"
#include "Shape.h"
#include "Type.h"


namespace LDG {
	class Tensor {
		public:
			Tensor() : shape_(), /*v(NULL),*/ device_type(CPU) {
			}

			Tensor(const Shape& shape) : shape_(shape)/*, v(NULL)*/ {
			}

			~Tensor() {
				/*
				if(device_type == CUDA && v != NULL)
					cudaFree(v);
				if(device_type == CPU && v != NULL)
					delete v;
				v = NULL;
				*/
			}

			/**
			 * Returns the shape of the Tensor.
			 * @return Shape of the Tensor.
			 */
			const Shape& shape() const {
				return shape_;
			}

			int device_type;
			//dtype *v;
			Shape shape_;

			const void* get_handle() const {
				return handle_.get();
			}

			std::shared_ptr<void> handle_;
	};

	typedef  Tensor* PTensor;
}

class IndexPtr {
	public:
		std::shared_ptr<void> ptr_;
		int size;

		IndexPtr() {}

		int* get_ptr() {
			return (int *)ptr_.get();
		}

};

class DtypePtr {
	public:
		std::shared_ptr<void> ptr_;
		int size;

		DtypePtr() {}

		dtype* get_ptr() {
			return (dtype *)ptr_.get();
		}
};

#endif // !LDG_TENSOR
