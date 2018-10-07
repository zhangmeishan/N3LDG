/*
 * SparseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "BaseParam.h"

// Notice: aux_square is an aux_squareiliary variable to help parameter updating
// The in-out dimension definiation is different with dense parameters.
class SparseParam : public BaseParam {
	public:
		//LDG::Tensor aux_square;
		//LDG::Tensor aux_mean;
		NRVec<bool> indexers;
		NRVec<int> last_update;

		vector<dtype> cpu_grad, cpu_aux_square, cpu_aux_mean, cpu_val;


		//LDG::Tensor cpu_grad;
		//LDG::Tensor cpu_aux_square;
		//LDG::Tensor cpu_val;

		//LDG::Tensor cpu_aux_mean;

		// allow sparse and dense parameters have different parameter initialization methods
		inline void initial(int outDim, int inDim) {
			//not in the aligned memory pool
			//val.init(outDim, inDim);
			dtype bound = sqrt(3.0 / (outDim));
			DEV->random_uniform(val, Shape({outDim, inDim}), -bound, bound);
			DEV->init(grad, Shape({outDim, inDim}));
			//DEV->init(aux_square, Shape({outDim, inDim}));
			//DEV->init(aux_mean, Shape({outDim, inDim}));

			int size = outDim * inDim;
			cpu_aux_square.resize(size);
			cpu_aux_mean.resize(size);
			for(int idx = 0; idx < size; idx++) {
				cpu_aux_square[idx] = 0;
				cpu_aux_mean[idx] = 0;
			}

			/*
			   cpu_grad.device_type = CPU;	
			   cpu_aux_square.device_type = CPU;	
			   cpu_val.device_type = CPU;	

			   cpu_grad.shape_ = grad.shape();
			   cpu_aux_square.shape_ = aux_square.shape();
			   cpu_val.shape_ = val.shape();

			   cpu_grad.v = new dtype[grad.shape().size()];
			   cpu_aux_square.v = new dtype[aux_square.shape().size()];
			   cpu_val.v = new dtype[val.shape().size()];

			   cpu_aux_mean.device_type = CPU;

			   cpu_aux_mean.shape_ = aux_mean.shape();

			   cpu_aux_mean.v = new dtype[aux_mean.shape().size()];
			 */
			//val.random(bound);
			//grad.init(outDim, inDim);
			//aux_square.init(outDim, inDim);
			//aux_mean.init(outDim, inDim);
			indexers.resize(inDim);
			indexers = false;
			last_update.resize(inDim);
			last_update = 0;
		}

		inline void clearGrad() {
			int inDim = indexers.size();
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				DEV->set_col(grad, index, 0);
				/*
				   for (int idx = 0; idx < grad.row; idx++) {
				   grad[index][idx] = 0;
				   }
				 */
			}

			indexers = false;
		}

		inline int outDim() {
			return val.shape().dims()[0];
			//return val.row;
		}

		inline int inDim() {
			return val.shape().dims()[1];
			//return val.col;
		}

		inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
			vector<int> id;
			int inDim = indexers.size();
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				id.push_back(index);
			}

			vector<dtype> g = DEV->to_vector(grad, id);
			vector<dtype> v = DEV->to_vector(val, id);
			int row = grad.shape().dims()[0];
			int offset = 0;

			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				for (int idx = 0; idx < row; idx++) {
					g[offset * row + idx] = g[offset * row + idx] + v[offset * row + idx] * reg;
					cpu_aux_square[index * row + idx] = cpu_aux_square[index * row + idx] + g[offset * row + idx] * g[offset * row + idx];
					v[offset * row + idx] = v[offset * row + idx] - g[offset * row + idx] * alpha / sqrt(cpu_aux_square[index * row + idx] + eps);
				}
				offset++;
			}

			DEV->set_cols(grad, id, g);
			DEV->set_cols(val, id, v);


			/*
			   vector<dtype> cpu_grad = DEV->to_vector(grad);
			//vector<dtype> cpu_aux_square = DEV->to_vector(aux_square);
			vector<dtype> cpu_val = DEV->to_vector(val);

			int row = grad.shape().dims()[0];
			for (int index = 0; index < inDim; index++) {
			if (!indexers[index]) continue;
			for (int idx = 0; idx < row; idx++) {
			cpu_grad[index * row + idx] = cpu_grad[index * row + idx] + cpu_val[index * row + idx] * reg;
			cpu_aux_square[index * row + idx] = cpu_aux_square[index * row + idx] + cpu_grad[index * row + idx] * cpu_grad[index * row + idx];
			cpu_val[index * row + idx] = cpu_val[index * row + idx] - cpu_grad[index * row + idx] * alpha / sqrt(cpu_aux_square[index * row + idx] + eps);
			}
			}
			DEV->set(grad, cpu_grad);
			//DEV->set(aux_square, cpu_aux_square);
			DEV->set(val, cpu_val);
			 */
			/*
			   int inDim = indexers.size();
			   for (int index = 0; index < inDim; index++) {
			   if (!indexers[index]) continue;
			   for (int idx = 0; idx < grad.row; idx++) {
			   grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
			   aux_square[index][idx] = aux_square[index][idx] + grad[index][idx] * grad[index][idx];
			   val[index][idx] = val[index][idx] - grad[index][idx] * alpha / sqrt(aux_square[index][idx] + eps);
			   }
			   }
			 */
		}

		inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {

			vector<int> id;
			int inDim = indexers.size();
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				id.push_back(index);
			}

			vector<dtype> g = DEV->to_vector(grad, id);
			vector<dtype> v = DEV->to_vector(val, id);

			dtype lr_t;
			int row = grad.shape().dims()[0];
			int offset = 0;
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				for (int idx = 0; idx < row; idx++) {
					g[offset * row + idx] = g[offset * row + idx] + v[offset * row + idx] * reg;
					cpu_aux_mean[index * row + idx] = belta1 * cpu_aux_mean[index * row + idx] + (1 - belta1) * g[offset * row + idx];
					cpu_aux_square[index * row + idx] = belta2 * cpu_aux_square[index * row + idx] + (1 - belta2) * g[offset * row + idx] * g[offset * row + idx];
					lr_t = alpha * sqrt(1 - pow(belta2, last_update[index] + 1)) / (1 - pow(belta1, last_update[index] + 1));
					v[offset * row + idx] = v[offset * row + idx] - cpu_aux_mean[index * row + idx] * lr_t / sqrt(cpu_aux_square[index * row + idx] + eps);
				}
				offset++;
				last_update[index]++;
			}

			DEV->set_cols(grad, id, g);
			DEV->set_cols(val, id, v);
			/*
			   cpu_grad = DEV->to_vector(grad);
			//cpu_aux_square = DEV->to_vector(aux_square);
			//cpu_aux_mean = DEV->to_vector(aux_mean);
			cpu_val = DEV->to_vector(val);

			dtype lr_t;
			//int inDim = indexers.size();
			int row = grad.shape().dims()[0];
			for (int index = 0; index < inDim; index++) {
			if (!indexers[index]) continue;
			for (int idx = 0; idx < row; idx++) {
			cpu_grad[index * row + idx] = cpu_grad[index * row + idx] + cpu_val[index * row + idx] * reg;
			cpu_aux_mean[index * row + idx] = belta1 * cpu_aux_mean[index * row + idx] + (1 - belta1) * cpu_grad[index * row + idx];
			cpu_aux_square[index * row + idx] = belta2 * cpu_aux_square[index * row + idx] + (1 - belta2) * cpu_grad[index * row + idx] * cpu_grad[index * row + idx];
			lr_t = alpha * sqrt(1 - pow(belta2, last_update[index] + 1)) / (1 - pow(belta1, last_update[index] + 1));
			cpu_val[index * row + idx] = cpu_val[index * row + idx] - cpu_aux_mean[index * row + idx] * lr_t / sqrt(cpu_aux_square[index * row + idx] + eps);
			}
			last_update[index]++;
			}
			DEV->set(grad, cpu_grad);
			//DEV->set(aux_square, cpu_aux_square);
			//DEV->set(aux_mean, cpu_aux_mean);
			DEV->set(val, cpu_val);
			 */

			/*
			   dtype lr_t;
			   int inDim = indexers.size();
			   for (int index = 0; index < inDim; index++) {
			   if (!indexers[index]) continue;
			   for (int idx = 0; idx < grad.row; idx++) {
			   grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
			   aux_mean[index][idx] = belta1 * aux_mean[index][idx] + (1 - belta1) * grad[index][idx];
			   aux_square[index][idx] = belta2 * aux_square[index][idx] + (1 - belta2) * grad[index][idx] * grad[index][idx];
			   lr_t = alpha * sqrt(1 - pow(belta2, last_update[index] + 1)) / (1 - pow(belta1, last_update[index] + 1));
			   val[index][idx] = val[index][idx] - aux_mean[index][idx] * lr_t / sqrt(aux_square[index][idx] + eps);
			   }
			   last_update[index]++;
			   }
			 */

		}

		inline void randpoint(int& idx, int &idy) {
			//select indexes randomly
			std::vector<int> idRows, idCols;
			idRows.clear();
			idCols.clear();
			int inDim = indexers.size();
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				idCols.push_back(index);
			}
			int row = val.shape().dims()[0];

			for (int i = 0; i < row; i++) {
				idRows.push_back(i);
			}

			random_shuffle(idRows.begin(), idRows.end());
			random_shuffle(idCols.begin(), idCols.end());

			idx = idCols[0];
			idy = idRows[0];
		}

		inline dtype squareGradNorm() {
			dtype sumNorm = 0.0;

			int inDim = indexers.size();
			int row = grad.shape().dims()[0];
			vector<int> id;
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				id.push_back(index);
			}
			vector<dtype> g= DEV->to_vector(grad, id);
			int size = g.size();
			for(int idx = 0; idx < size; idx++) {
				sumNorm += g[idx] * g[idx];
			}

			/*
			   int inDim = indexers.size();
			   for (int index = 0; index < inDim; index++) {
			   if (!indexers[index]) continue;
			   for (int idx = 0; idx < val.row; idx++) {
			   sumNorm += grad[index][idx] * grad[index][idx];
			   }
			   }
			 */

			return sumNorm;
		}

		inline void rescaleGrad(dtype scale) {
			int inDim = indexers.size();
			vector<int> id;
			for (int index = 0; index < inDim; index++) {
				if (!indexers[index]) continue;
				id.push_back(index);
			}
			vector<dtype> cpu_grad = DEV->to_vector(grad, id);
			int max_size = cpu_grad.size();
			for(int idx = 0; idx < max_size; idx++) {
				cpu_grad[idx] = cpu_grad[idx] * scale;
			}
			DEV->set_cols(grad, id, cpu_grad);

			/*
			   vector<dtype> cpu_grad = DEV->to_vector(grad);
			   int row = grad.shape().dims()[0];

			   int inDim = indexers.size();
			   for (int index = 0; index < inDim; index++) {
			   if (!indexers[index]) continue;
			   for (int idx = 0; idx < row; idx++) {
			   cpu_grad[index * row + idx] = cpu_grad[index * row + idx] * scale;
			   }
			   }

			   DEV->set(grad, cpu_grad);
			 */

			/*
			   int inDim = indexers.size();
			   for (int index = 0; index < inDim; index++) {
			   if (!indexers[index]) continue;
			   for (int idx = 0; idx < val.row; idx++) {
			   grad[index][idx] = grad[index][idx] * scale;
			   }
			   }
			 */
		}

		//inline void value(const int& featId, Tensor1D& out) {
		inline void value(const int& featId, LDG::Tensor& out) {
			//if (out.dim != val.row) {
			if (out.shape().dims()[0]!= val.shape().dims()[0]) {
				std::cout << "warning: output dim not equal lookup param dim." << std::endl;
			}

			/*
			   for (int idx = 0; idx < val.row; idx++) {
			   out[idx] = val[featId][idx];
			   }
			 */

			DEV->get_col(val, featId, out);
		}

		//inline void value(const vector<int>& featIds, Tensor1D& out) {
		inline void value(const vector<int>& featIds, LDG::Tensor& out) {
			/*
			   if (out.dim != val.row) {
			   std::cout << "warning: output dim not equal lookup param dim." << std::endl;
			   }

			   int featNum = featIds.size();
			   int featId;
			   for (int i = 0; i < featNum; i++) {
			   featId = featIds[i];
			   for (int idx = 0; idx < val.row; idx++) {
			   out[idx] += val[featId][idx];
			   }
			   }
			 */
		}

		inline void loss(const int& featId, const LDG::Tensor& loss) {
			int row = val.shape().dims()[0];
			int loss_dim = loss.shape().dims()[0];
			if (loss_dim != row) {
				std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
			}
			indexers[featId] = true;
			DEV->Fadd_col(grad, loss, featId);
			/*
			   if (loss.dim != val.row) {
			   std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
			   }
			   indexers[featId] = true;
			   for (int idx = 0; idx < val.row; idx++) {
			   grad[featId][idx] += loss[idx];
			   }
			 */
		}

		inline void loss(const vector<int>& featIds, const LDG::Tensor& loss) {
			/*
			   if (loss.dim != val.row) {
			   std::cout << "warning: loss dim not equal lookup param dim." << std::endl;
			   }
			   int featNum = featIds.size();
			   int featId;
			   for (int i = 0; i < featNum; i++) {
			   featId = featIds[i];
			   indexers[featId] = true;
			   for (int idx = 0; idx < val.row; idx++) {
			   grad[featId][idx] += loss[idx];
			   }
			   }
			 */
		}

		inline void save(std::ofstream &os)const {
			/*
			   val.save(os);
			   aux_square.save(os);
			   aux_mean.save(os);
			   os << val.col << std::endl;
			   for (int idx = 0; idx < val.col; idx++) {
			   os << last_update[idx] << std::endl;
			   }
			 */
		}

		inline void load(std::ifstream &is) {
			/*
			   val.load(is);
			   aux_square.load(is);
			   aux_mean.load(is);
			   int curInDim;
			   is >> curInDim;
			   last_update.resize(curInDim);
			   for (int idx = 0; idx < curInDim; idx++) {
			   is >> last_update[idx];
			   }
			 */
		}

		};

#endif /* SPARSEPARAM_H_ */
