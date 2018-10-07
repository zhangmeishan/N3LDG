#ifndef ATOMICIOP_H_
#define ATOMICIOP_H_

/*
 *  AtomicOP.h:
 *  a list of atomic operations
 *
 *  Created on: June 11, 2017
 *      Author: yue_zhang(suda), mszhang
 */

/*
   ActivateNode
   TanhNode
   SigmoidNode
   ReluNode
   IndexNode
   PSubNode
   PDotNode
 */

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class TanhNode :public Node {
	public:
		PNode in;
		LDG::Tensor lx;

	public:
		TanhNode() : Node() {
			in = NULL;
			node_type = "tanh";
		}

		~TanhNode() {
			in = NULL;
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;
		}

	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}

	public:
		/*
		   inline void compute() {
		   DEV->unaryExp(in->val, val, DEV, &Device::Ftanh);
		   }

		   void backward() {
		   LDG::Tensor v_d;
		   DEV->init(v_d, in->val.shape());
		   DEV->binaryExp(in->val, val, v_d, DEV, &Device::Dtanh);

		   LDG::Tensor temp_loss;
		   DEV->init(temp_loss, in->loss.shape());
		   DEV->Fmatmul(loss, v_d, temp_loss);

		   DEV->Fadd_inplace(in->loss, temp_loss);
		   }
		 */

	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			return result;
		}
};

class TanhExecute :public Execute {
	public:
		vector<LDG::PTensor> vec_x, vec_val;
		bool bTrain;

	public:
		~TanhExecute() {
		}

	public:
		inline void  forward() {
			int count = batch.size();

			vec_x.clear();
			vec_val.clear();
			TanhNode* ptr = (TanhNode*)batch[0];
			drop_value = ptr->drop_value;
			for (int idx = 0; idx < count; idx++) {
				TanhNode* ptr = (TanhNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
				vec_val.push_back(&ptr->val);
				ptr->degree = -1;
			}

			DEV->unaryExp(vec_x, vec_val, DEV, &Device::Ftanh);
			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
		}


		inline void backward() {
			int count = batch.size();
			vector<LDG::PTensor> vec_loss, vec_in_loss;
			vector<LDG::PTensor> vec_lx;
			for (int idx = 0; idx < count; idx++) {
				TanhNode* ptr = (TanhNode*)batch[idx];
				vec_loss.push_back(&ptr->loss);
				vec_in_loss.push_back(&ptr->in->loss);
				vec_lx.push_back(&ptr->lx);
			}

			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}

			DEV->binaryExp(vec_x, vec_val, vec_lx, DEV, &Device::Dtanh);
			DEV->Fmultiply_inplace(vec_lx, vec_loss);
			DEV->Fadd_inplace(vec_in_loss, vec_lx);
		}
};

inline PExecute TanhNode::generate(bool bTrain) {
	TanhExecute* exec = new TanhExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
};

class PDotNode : public Node {
	public:
		PNode in1, in2;
	public:
		PDotNode() : Node() {
			in1 = NULL;
			in2 = NULL;
			dim = 1;
			node_type = "point-dot";
		}
	public:
		virtual inline void clearValue() {
			Node::clearValue();
			in1 = NULL;
			in2 = NULL;
		}

		//can not be dropped since the output is a scalar
		inline void init(int ndim, dtype dropout) {
			dim = 1;
			Node::init(dim, -1);
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
		inline void compute() {
			DEV->zero(val);	
			DEV->Fmatmul(in1->val, in2->val, val, true, false);
			//val[0] = 0.0;
			//for (int idx = 0; idx < in1->dim; idx++) {
			//val[0] += in1->val[idx] * in2->val[idx];
			//}
		}

		void backward() {
			//LDG::Tensor cpu_loss;
			//cpu_loss.device_type == CPU;
			//cpu_loss.shape_ = loss.shape();
			//cpu_loss.v = new dtype[loss.shape().size()];
			//DEV->to_cpu(loss, cpu_loss);
			//vector<dtype> loss_val = DEV->to_vector(loss);

			LDG::Tensor temp_loss1, temp_loss2;
			//DEV->Fmultiply_scalar(in2->val, loss_val[0], temp_loss1);
			//DEV->Fmultiply_scalar(in1->val, loss_val[0], temp_loss2);

			DEV->Fmultiply_scalar(in2->val, loss, temp_loss1);
			DEV->Fmultiply_scalar(in1->val, loss, temp_loss2);

			DEV->Fadd_inplace(in1->loss, temp_loss1);
			DEV->Fadd_inplace(in2->loss, temp_loss2);

			//DEV->Fadd(in1->loss, temp_loss1, in1->loss);
			//DEV->Fadd(in2->loss, temp_loss2, in2->loss);
		}

	public:
		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			return Node::typeEqual(other);
		}

		inline PExecute generate(bool bTrain);
};

class PDotExecute :public Execute {
	public:
		bool bTrain;
		vector<LDG::PTensor> vec_in_val1, vec_in_val2, vec_val;
	public:
		inline void  forward() {
			int count = batch.size();
			vec_in_val1.resize(count);
			vec_in_val2.resize(count);
			vec_val.resize(count);
			PDotNode* ptr = (PDotNode*)batch[0];
			drop_value = ptr->drop_value;
			for (int idx = 0; idx < count; idx++) {
				PDotNode* ptr = (PDotNode*)batch[idx];
				vec_in_val1[idx] = &(ptr->in1->val);
				vec_in_val2[idx] = &(ptr->in2->val);
				vec_val[idx] = &(ptr->val);
				ptr->degree = -1;
			}
			DEV->Fdot(vec_in_val1, vec_in_val2, vec_val, true, false);

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
		}

		inline void backward() {
			int count = batch.size();
			vector<LDG::PTensor> vec_loss, vec_in_loss1, vec_in_loss2;
			for (int idx = 0; idx < count; idx++) {
				PDotNode* ptr = (PDotNode*)batch[idx];
				vec_loss.push_back(&ptr->loss);
				vec_in_loss1.push_back(&ptr->in1->loss);
				vec_in_loss2.push_back(&ptr->in2->loss);
			}
			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}
			DEV->Ddot(vec_in_loss2, vec_in_val1, vec_loss);
			DEV->Ddot(vec_in_loss1, vec_in_val2, vec_loss);
		}
};


inline PExecute PDotNode::generate(bool bTrain) {
	PDotExecute* exec = new PDotExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}

#endif
