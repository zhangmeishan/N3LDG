/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

//#include "Eigen/Dense"
#include "BaseParam.h"



// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
	public:
		LDG::Tensor aux_square;
		LDG::Tensor aux_mean;
		int iter;

		//LDG::Tensor cpu_grad;

		LDG::Tensor v_r;
		LDG::Tensor grad_square;
		LDG::Tensor aux_eps;
		LDG::Tensor aux_sqrt;
		LDG::Tensor grad_alpha;
		LDG::Tensor grad_aux;

		LDG::Tensor belta_aux_mean;
		LDG::Tensor belta_grad;
		LDG::Tensor belta_aux_square;
		LDG::Tensor belta_grad_square;
		LDG::Tensor aux_square_eps;
		LDG::Tensor aux_square_eps_sqrt;
		LDG::Tensor aux_mean_lrt;
		LDG::Tensor val_delta;

		// allow sparse and dense parameters have different parameter initialization methods
		inline void initial(int outDim, int inDim) {
			//val.init(outDim, inDim);
			//grad.init(outDim, inDim);
			//aux_square.init(outDim, inDim);
			//aux_mean.init(outDim, inDim);
			//DEV->malloc(val, Shape({outDim, inDim}));
			DEV->init(grad, Shape({outDim, inDim}));
			DEV->init(aux_square, Shape({outDim, inDim}));
			DEV->init(aux_mean, Shape({outDim, inDim}));

			dtype bound = sqrt(6.0 / (outDim + inDim + 1));
			//val.random(bound);
			DEV->random_uniform(val, Shape({outDim, inDim}), -bound, bound);

			DEV->init(v_r, val.shape()); 
			DEV->init(grad_square, grad.shape());
			DEV->init(aux_eps, aux_square.shape());
			DEV->init(aux_sqrt, aux_square.shape());
			DEV->init(grad_alpha, grad.shape());
			DEV->init(grad_aux, grad.shape());

			DEV->init(belta_aux_mean, aux_mean.shape());
			DEV->init(belta_grad, grad.shape());
			DEV->init(belta_aux_square, aux_square.shape());
			DEV->init(belta_grad_square, grad.shape());            
			DEV->init(aux_square_eps, aux_square.shape());
			DEV->init(aux_square_eps_sqrt, aux_square.shape());
			DEV->init(aux_mean_lrt, aux_mean.shape());
			DEV->init(val_delta, val.shape());
			iter = 0;

			//cpu_grad.device_type = CPU;	
			//cpu_grad.shape_ = grad.shape();
			//cpu_grad.v = new dtype[grad.shape().size()];
		}

		inline int outDim() {
			//return val.row;
			return val.shape().dims()[0];
		}

		inline int inDim() {
			//return val.col;
			return val.shape().dims()[1];
		}

		inline void clearGrad() {
			//grad.zero();
			DEV->zero(grad);
		}

		inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
			if (outDim() > 1 && inDim() > 1) {
				DEV->Fmultiply_scalar(val, reg, v_r);
				DEV->Fadd_inplace(grad, v_r);
				//DEV->Fadd(grad, v_r, grad);
				//grad.vec() = grad.vec() + val.vec() * reg;
			}
			DEV->Fsquare(grad, grad_square);
			DEV->Fadd_inplace(aux_square, grad_square);
			//DEV->Fadd(aux_square, grad_square, aux_square);
			//aux_square.vec() = aux_square.vec() + grad.vec().square();

			DEV->Fadd_scalar(aux_square, eps, aux_eps);
			DEV->Fsqrt(aux_eps, aux_sqrt);

			DEV->Fmultiply_scalar(grad, alpha, grad_alpha);

			DEV->Fdivide(grad_alpha, aux_sqrt, grad_aux);

			DEV->Fsubtract_inplace(val, grad_aux);

			//DEV->Fsubtract(val, grad_aux, val);
			//val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
		}

		inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
			if (outDim() > 1 && inDim() > 1) {
				DEV->Fmultiply_scalar(val, reg, v_r);
				DEV->Fadd_inplace(grad, v_r);
				//DEV->Fadd(grad, v_r, grad);
			}

			DEV->Fmultiply_scalar(aux_mean, belta1, belta_aux_mean);

			DEV->Fmultiply_scalar(grad, 1 - belta1, belta_grad);

			DEV->Fadd(belta_aux_mean, belta_grad, aux_mean);

			DEV->Fmultiply_scalar(aux_square, belta2, belta_aux_square);

			DEV->Fsquare(grad, grad_square);

			DEV->Fmultiply_scalar(grad_square, (1 - belta2), belta_grad_square);

			DEV->Fadd(belta_aux_square, belta_grad_square, aux_square);

			dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));

			DEV->Fadd_scalar(aux_square, eps, aux_square_eps);

			DEV->Fsqrt(aux_square_eps, aux_square_eps_sqrt);

			DEV->Fmultiply_scalar(aux_mean, lr_t, aux_mean_lrt);

			DEV->Fdivide(aux_mean_lrt, aux_square_eps_sqrt, val_delta);

			DEV->Fsubtract_inplace(val, val_delta);

			//DEV->Fsubtract(val, val_delta, val);


			iter++;
			/*
			   if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
			   aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
			   aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
			   dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
			   val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
			   iter++;
			 */
		}

		inline void randpoint(int& idx, int &idy) {
			//select indexes randomly
			std::vector<int> idRows, idCols;
			idRows.clear();
			idCols.clear();
			int dim0 = val.shape().dims()[0];
			int dim1 = val.shape().dims()[1];
			for (int i = 0; i < dim0; i++)
				idRows.push_back(i);
			for (int i = 0; i < dim1; i++)
				idCols.push_back(i);

			random_shuffle(idRows.begin(), idRows.end());
			random_shuffle(idCols.begin(), idCols.end());

			idy = idRows[0];
			idx = idCols[0];
		}

		inline dtype squareGradNorm() {
			dtype sumNorm = 0.0;

			//DEV->to_cpu(grad, cpu_grad);
			vector<dtype> vec_grad = DEV->to_vector(grad);
			int size = grad.shape().size();
			for (int i = 0; i < size; i++) {
				sumNorm += vec_grad[i] * vec_grad[i];
			}
			return sumNorm;
		}

		inline void rescaleGrad(dtype scale) {
			//grad.vec() = grad.vec() * scale;
			DEV->Fmultiply_scalar_inplace(grad, scale);
			//DEV->Fmultiply_scalar(grad, scale, grad);
		}

		inline void save(std::ofstream &os)const {
			/*
			   val.save(os);
			   aux_square.save(os);
			   aux_mean.save(os);
			   os << iter << endl;
			 */
		}

		inline void load(std::ifstream &is) {
			/*
			   val.load(is);
			   aux_square.load(is);
			   aux_mean.load(is);
			   is >> iter;
			 */
		}
};

#endif /* PARAM_H_ */
