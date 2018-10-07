#ifndef UNIOP_H_
#define UNIOP_H_

/*
 *  UniOP.h:
 *  a simple feed forward neural operation, unary input.
 *
 *  Created on: Apr 22, 2017
 *      Author: mszhang
 */


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class UniParams {
	public:
		Param W;
		Param b;
		bool bUseB;

	public:
		UniParams() {
			bUseB = true;
		}

		inline void exportAdaParams(ModelUpdate& ada) {
			ada.addParam(&W);
			if (bUseB) {
				ada.addParam(&b);
			}
		}

		inline void initial(int nOSize, int nISize, bool useB = true) {
			W.initial(nOSize, nISize);
			bUseB = useB;
			if (bUseB) {
				b.initial(nOSize, 1);
			}
		}

		inline void save(std::ofstream &os) const {
			os << bUseB << std::endl;
			W.save(os);
			if (bUseB) {
				b.save(os);
			}
		}

		inline void load(std::ifstream &is) {
			is >> bUseB;
			W.load(is);
			if (bUseB) {
				b.load(is);
			}
		}

};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)

class UniNode : public Node {
	public:
		PNode in;
		LDG::Tensor ty, lty, dy, lx;
		UniParams* param;
		void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
		void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		//void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		//dtype(*activate)(const dtype&);   // activation function
		//dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function


	public:
		UniNode() : Node() {
			in = NULL;
			//activate = ftanh;
			//derivate = dtanh;
			activate = &Device::Ftanh;
			derivate = &Device::Dtanh;
			param = NULL;
			node_type = "uni";
		}

		~UniNode() {
			in = NULL;
		}


		inline void setParam(UniParams* paramInit) {
			param = paramInit;
		}

		void init(int ndim, dtype dropout) {
			Node::init(ndim, dropout);
			DEV->init(ty, Shape({ndim, 1}));
			DEV->init(lty, Shape({ndim, 1}));
			//DEV->init(dy, Shape({ndim, 1}));

			//int inDim = param->W.val.shape().dims()[1];
			//DEV->init(lx, Shape({inDim, 1}));
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;
		}

		// define the activate function and its derivation form
		inline void setFunctions(void (Device::*f)(const LDG::Tensor&, LDG::Tensor&),
				void (Device::*f_deri)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&)) {
			activate = f;
			derivate = f_deri;
		}

	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}


	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			if (!result) return false;

			UniNode* conv_other = (UniNode*)other;
			if (param != conv_other->param) {
				return false;
			}
			if (activate != conv_other->activate || derivate != conv_other->derivate) {
				return false;
			}

			return true;
		}

};

class UniExecute :public Execute {
	public:
		//Tensor2D x, ty, y, b;

		vector<LDG::PTensor> vec_x;
		vector<LDG::PTensor> vec_b;
		vector<LDG::PTensor> vec_ty;
		vector<LDG::PTensor> vec_val;

		int inDim, outDim;
		UniParams* param;
		void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
		void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
		//dtype(*activate)(const dtype&);   // activation function
		//dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
		bool bTrain;

		LDG::Tensor x, b, ty, val;


	public:
		inline void  forward() {
			int count = batch.size();
			vec_x.clear();
			vec_b.clear();
			vec_ty.clear();
			vec_val.clear();

			UniNode* ptr = (UniNode*)batch[0];
			drop_value = ptr->drop_value;

			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
				if (param->bUseB) {
					vec_b.push_back(&param->b.val);
				}

				vec_ty.push_back(&ptr->ty);
				vec_val.push_back(&ptr->val);
				ptr->degree = -1;
			}

			DEV->concat(vec_x, x);
			DEV->concat(vec_b, b);
			DEV->Fmatmul(param->W.val, x, ty);
			//ty.mat() = param->W.val.mat() * x.mat();

			if (param->bUseB) {
				DEV->Fadd_inplace(ty, b);
				//DEV->Fadd(vec_ty, vec_b, vec_ty);
				//ty.vec() = ty.vec() + b.vec();
			}

			//y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

			DEV->unaryExp(ty, val, DEV, activate);
			DEV->unconcat(val, vec_val);

			//for (int idx = 0; idx < count; idx++) {
			//UniNode* ptr = (UniNode*)batch[idx];
			//DEV->show_val(ptr->val);
			//}
			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}

			/*
			   for (int idx = 0; idx < count; idx++) {
			   UniNode* ptr = (UniNode*)batch[idx];
			   ptr->forward_drop(bTrain);
			   }
			 */
		}

		inline void backward() {
			int count = batch.size();

			vector<LDG::PTensor> vec_loss;
			vector<LDG::PTensor> vec_dy;
			vector<LDG::PTensor> vec_lty;
			vector<LDG::PTensor> vec_lx;
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				//ptr->backward_drop();

				vec_loss.push_back(&ptr->loss);
				vec_dy.push_back(&ptr->dy);
				vec_lty.push_back(&ptr->lty);
				vec_lx.push_back(&ptr->lx);
			}

			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}

			LDG::Tensor dy, ly, lty;
			DEV->concat(vec_loss, ly);
			DEV->binaryExp(ty, val, dy, DEV, derivate);
			DEV->Fmultiply(ly, dy, lty);
			//DEV->unconcat(lty, vec_lty);


			//lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

			//LDG::Tensor wg;
			//DEV->malloc(wg, param->W.grad.shape());
			DEV->Dmatmul(lty, x, param->W.grad, false, true);
			//DEV->Fadd_inplace(param->W.grad, wg);
			//DEV->Fadd(param->W.grad, wg, param->W.grad);

			//param->W.grad.mat() += lty.mat() * x.mat().transpose();
			if (param->bUseB) {
				DEV->Dadd_inplace(param->b.grad, lty);
				//DEV->Fadd(param->b.grad, vec_lty, param->b.grad);
			}
			vector<LDG::PTensor> vec_in_loss;
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				vec_in_loss.push_back(&ptr->in->loss);
			}

			DEV->Dmatmul(param->W.val, lty, vec_in_loss, true, false);


			//DEV->Fadd_inplace(vec_in_loss, vec_lx);
			//DEV->Fadd(vec_in_loss, vec_lx, vec_in_loss);
		}
};

inline PExecute UniNode::generate(bool bTrain) {
	UniExecute* exec = new UniExecute();
	exec->batch.push_back(this);
	exec->inDim = param->W.inDim();
	exec->outDim = param->W.outDim();
	exec->param = param;
	exec->activate = activate;
	exec->derivate = derivate;
	exec->bTrain = bTrain;
	return exec;
}

class LinearNode : public Node {
	public:
		PNode in;
		UniParams* param;
		LDG::Tensor lx;

	public:
		LinearNode() : Node() {
			in = NULL;
			param = NULL;
			node_type = "linear";
		}


		inline void setParam(UniParams* paramInit) {
			param = paramInit;
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;
		}


		//void init(int ndim, dtype dropout) {
		//Node::init(ndim, dropout);
		//int inDim = param->W.val.shape().dims()[1];

		//DEV->init(lx, Shape({inDim, 1}));
		//}


	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}

	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			if (!result) return false;
			LinearNode* conv_other = (LinearNode*)other;
			if (param != conv_other->param) {
				return false;
			}

			return true;
		}

};

class LinearExecute :public Execute {
	public:
		vector<LDG::PTensor> vec_x;
		vector<LDG::PTensor> vec_val;

		vector<LDG::PTensor> vec_loss;
		vector<LDG::PTensor> vec_lx;
		vector<LDG::PTensor> vec_in_loss;
		//	Tensor2D x, y;
		int inDim, outDim, count;
		UniParams* param;
		bool bTrain;

		LDG::Tensor x, val;

	public:
		inline void  forward() {
			count = batch.size();
			//x.init(inDim, count);
			//y.init(outDim, count);

			LinearNode* ptr = (LinearNode*)batch[0];
			drop_value = ptr->drop_value;

			vec_x.resize(count);
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_x[idx] = (&ptr->in->val);
				ptr->degree = -1;
			}
			vec_val.resize(count);
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_val[idx] = (&ptr->val);
			}

			DEV->concat(vec_x, x);
			DEV->Fmatmul(param->W.val, x, val);
			//y.mat() = param->W.val.mat() * x.mat();

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
			DEV->unconcat(val, vec_val);
			/*
			   for (int idx = 0; idx < count; idx++) {
			   LinearNode* ptr = (LinearNode*)batch[idx];
			   ptr->forward_drop(bTrain);
			   }
			 */
		}

		inline void backward() {
			//	Tensor2D lx, ly;
			//	lx.init(inDim, count);
			//	ly.init(outDim, count);

			vec_loss.resize(count);
			vec_lx.resize(count);
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				//ptr->backward_drop();
				vec_loss[idx] = (&ptr->loss);
				vec_lx[idx] = (&ptr->lx);
				//for (int idy = 0; idy < outDim; idy++) {
				//ly[idx][idy] = ptr->loss[idy];
				//}
			}
			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}

			//LDG::Tensor wg;
			//DEV->malloc(wg, param->W.grad.shape());
			LDG::Tensor loss;
			DEV->concat(vec_loss, loss);
			DEV->Dmatmul(loss, x, param->W.grad, false, true);
			//DEV->Fadd_inplace(param->W.grad, wg);
			//DEV->Fadd(param->W.grad, wg, param->W.grad);
			//param->W.grad.mat() += ly.mat() * x.mat().transpose();
			vec_in_loss.resize(count);

			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_in_loss[idx] = (&ptr->in->loss);
			}

			DEV->Dmatmul(param->W.val, loss, vec_in_loss, true, false);

			//lx.mat() += param->W.val.mat().transpose() * ly.mat();


			//DEV->Fadd_inplace(vec_in_loss, vec_lx);
			//DEV->Fadd(vec_in_loss, vec_lx, vec_in_loss);
		}
};

inline PExecute LinearNode::generate(bool bTrain) {
	LinearExecute* exec = new LinearExecute();
	exec->batch.push_back(this);
	exec->inDim = param->W.inDim();
	exec->outDim = param->W.outDim();
	exec->param = param;
	exec->bTrain = bTrain;
	return exec;
}

#endif /* UNIOP_H_ */
