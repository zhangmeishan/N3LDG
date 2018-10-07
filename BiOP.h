#ifndef BIOP_H_
#define BIOP_H_

/*
 *  BiOP.h:
 *  a simple feed forward neural operation, binary input.
 *
 *  Created on: June 11, 2017
 *      Author: mszhang
 */


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class BiParams {
	public:
		Param W1;
		Param W2;
		Param b;

		bool bUseB;

	public:
		BiParams() {
			bUseB = true;
		}

		inline void exportAdaParams(ModelUpdate& ada) {
			ada.addParam(&W1);
			ada.addParam(&W2);
			if (bUseB) {
				ada.addParam(&b);
			}
		}

		inline void initial(int nOSize, int nISize1, int nISize2, bool useB = true) {
			W1.initial(nOSize, nISize1);
			W2.initial(nOSize, nISize2);
			bUseB = useB;
			if (bUseB) {
				b.initial(nOSize, 1);
			}
		}

		inline void save(std::ofstream &os) const {
			os << bUseB << std::endl;
			W1.save(os);
			W2.save(os);
			if (bUseB) {
				b.save(os);
			}
		}

		inline void load(std::ifstream &is) {
			is >> bUseB;
			W1.load(is);
			W2.load(is);
			if (bUseB) {
				b.load(is);
			}
		}

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class BiNode : public Node {
	public:
		PNode in1, in2;
		LDG::Tensor ty1, ty2, ty, y, dy, lty;
		LDG::Tensor lx1, lx2;
		BiParams* param;
		void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
		//void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		//dtype(*activate)(const dtype&);   // activation function
		//dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function


	public:
		BiNode() : Node() {
			in1 = in2 = NULL;
			activate = &Device::Ftanh;
			derivate = &Device::Dtanh;
			//activate = ftanh;
			//derivate = dtanh;
			param = NULL;
			node_type = "bi";
		}

		~BiNode() {
			in1 = in2 = NULL;
		}

		void init(int ndim, dtype dropout) {
			Node::init(ndim, dropout);
			//DEV->init(ty1, Shape({ndim, 1}));
			//DEV->init(ty2, Shape({ndim, 1}));
			DEV->init(ty, Shape({ndim, 1}));
			//DEV->init(y, Shape({ndim, 1}));
			DEV->init(lty, Shape({ndim, 1}));
			//DEV->init(dy, Shape({ndim, 1}));

			//int inDim1 = param->W1.val.shape().dims()[1];
			//DEV->init(lx1, Shape({inDim1, 1}));

			//int inDim2 = param->W2.val.shape().dims()[1];
			//DEV->init(lx2, Shape({inDim2, 1}));
		}

		inline void setParam(BiParams* paramInit) {
			param = paramInit;
		}

		inline void clearValue() {
			Node::clearValue();
			in1 = in2 = NULL;
		}

		// define the activate function and its derivation form
		inline void setFunctions(void (Device::*f)(const LDG::Tensor&, LDG::Tensor&),
				void (Device::*f_deri)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&)) {
			activate = f;
			derivate = f_deri;
		}


	public:
		void forward(Graph *cg, PNode x1, PNode x2) {
			in1 = x1;
			in2 = x2;
			degree = 0;
			in1->addParent(this);
			in2->addParent(this);
			cg->addNode(this);
		}


	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			if (!result) return false;

			BiNode* conv_other = (BiNode*)other;
			if (param != conv_other->param) {
				return false;
			}
			if (activate != conv_other->activate || derivate != conv_other->derivate) {
				return false;
			}

			return true;
		}

};



class BiExecute :public Execute {
	public:
		vector<vector<LDG::PTensor> > vec_vec_x;
		vector<LDG::PTensor> vec_x1, vec_x2, vec_b, vec_ty1, vec_ty2, vec_ty, vec_val, vec_dy, vec_lty;

		vector<LDG::PTensor> vec_loss;

		vector<LDG::PTensor> vec_lx1, vec_lx2;

		vector<LDG::PTensor> vec_in_loss1, vec_in_loss2;

		int inDim1, inDim2, outDim;
		LDG::Tensor x1, x2, b, ty, val;
		BiParams* param;

		void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
		//void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		//dtype(*activate)(const dtype&);   // activation function
		//dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
		bool bTrain;

	public:
		~BiExecute() {
			param = NULL;
			activate = NULL;
			derivate = NULL;
			inDim1 = inDim2 = outDim = 0;
		}


	public:
		inline void  forward() {
			int count = batch.size();
			vec_x1.resize(count);
			vec_x2.resize(count);
			vec_b.resize(count);

			vec_ty1.resize(count);
			vec_ty2.resize(count);
			vec_ty.resize(count);
			vec_val.resize(count);

			vec_vec_x.resize(count);
			BiNode* ptr = (BiNode*)batch[0];
			drop_value = ptr->drop_value;

			for (int idx = 0; idx < count; idx++) {
				BiNode* ptr = (BiNode*)batch[idx];
				vec_x1[idx] = (&ptr->in1->val);
				vec_x2[idx] = (&ptr->in2->val);

				vector<LDG::PTensor> vec_x;
				vec_x.push_back(&ptr->in1->val);
				vec_x.push_back(&ptr->in2->val);
				vec_vec_x[idx]= vec_x;
				ptr->degree = -1;


				vec_ty1[idx] = (&ptr->ty1);
				vec_ty2[idx] = (&ptr->ty2);
				vec_ty[idx] = (&ptr->ty);
				vec_val[idx] = (&ptr->val);
				if (param->bUseB) {
					vec_b[idx] = (&param->b.val);
				}
			}

			DEV->concat(vec_x1, x1);
			DEV->concat(vec_x2, x2);
			DEV->concat(vec_b, b);

			LDG::Tensor ty1, ty2;
			DEV->Fmatmul(param->W1.val, x1, ty1);
			DEV->Fmatmul(param->W2.val, x2, ty2);
			DEV->Fadd(ty1, ty2, ty);

			if (param->bUseB) {
				DEV->Fadd_inplace(ty, b);
			}


			DEV->unaryExp(ty, val, DEV, activate);
			DEV->unconcat(val, vec_val);

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
			/*
			   for (int idx = 0; idx < count; idx++) {
			   BiNode* ptr = (BiNode*)batch[idx];
			   ptr->forward_drop(bTrain);
			   }
			 */
		}

		inline void backward() {
			int count = batch.size();

			vec_loss.resize(count);
			vec_dy.resize(count);
			vec_lty.resize(count);
			for (int idx = 0; idx < count; idx++) {
				BiNode* ptr = (BiNode*)batch[idx];
				//ptr->backward_drop();
				vec_loss[idx] = (&ptr->loss);
				vec_dy[idx] = (&ptr->dy);
				vec_lty[idx] = (&ptr->lty);
			}

			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}

			LDG::Tensor loss, dy, lty;
			DEV->concat(vec_loss, loss);
			DEV->binaryExp(ty, val, dy, DEV, derivate);
			DEV->Fmultiply(loss, dy, lty);

			DEV->Dmatmul(lty, x1, param->W1.grad, false, true);
			DEV->Dmatmul(lty, x2, param->W2.grad, false, true);

			if (param->bUseB) {
				DEV->Dadd_inplace(param->b.grad, lty);
			}

			vec_lx1.resize(count); 
			vec_lx2.resize(count);
			vec_in_loss1.resize(count);
			vec_in_loss2.resize(count);
			for (int idx = 0; idx < count; idx++) {
				BiNode* ptr = (BiNode*)batch[idx];
				vec_lx1[idx] = (&ptr->lx1);
				vec_lx2[idx] = (&ptr->lx2);
				vec_in_loss1[idx] = (&ptr->in1->loss);
				vec_in_loss2[idx] = (&ptr->in2->loss);
			}

			DEV->Dmatmul(param->W1.val, lty, vec_in_loss1, true, false);
			DEV->Dmatmul(param->W2.val, lty, vec_in_loss2, true, false);
		}
};

inline PExecute BiNode::generate(bool bTrain) {
	BiExecute* exec = new BiExecute();
	exec->batch.push_back(this);
	exec->inDim1 = param->W1.inDim();
	exec->inDim2 = param->W2.inDim();
	exec->outDim = param->W1.outDim();
	exec->param = param;
	exec->activate = activate;
	exec->derivate = derivate;
	exec->bTrain = bTrain;
	return exec;
}

#endif /* BIOP_H_ */
