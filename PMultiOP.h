#ifndef PMultiOP
#define PMultiOP

/*
 *  PMultiOP.h:
 *  pointwise multiplication
 *
 *  Created on: Apr 21, 2017
 *      Author: mszhang
 */

//#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PMultiNode : public Node {
	public:
		PNode in1, in2;

		LDG::Tensor lx1, lx2;
	public:
		PMultiNode() : Node() {
			in1 = NULL;
			in2 = NULL;
			node_type = "point-multiply";
		}
	public:
		virtual inline void clearValue() {
			Node::clearValue();
			in1 = NULL;
			in2 = NULL;
		}

	public:
		void forward(Graph *cg, PNode x1, PNode x2) {
			in1 = x1;
			in2 = x2;
			degree = 0;
			x1->addParent(this);
			x2->addParent(this);
			cg->addNode(this);
		}

	public:
		/*
		   inline void compute() {
		   DEV->Fmultiply(in1->val, in2->val, val);
	//val.vec() = in1->val.vec() * in2->val.vec();
	}

	void backward() {
	LDG::Tensor temp_loss1, temp_loss2; 
	DEV->Fmultiply(loss, in2->val, temp_loss1);
	DEV->Fadd_inplace(in1->loss, temp_loss1);
		//DEV->Fadd(in1->loss, temp_loss1, in1->loss);

		DEV->Fmultiply(loss, in1->val, temp_loss2);
		DEV->Fadd_inplace(in2->loss, temp_loss2);
		//DEV->Fadd(in2->loss, temp_loss2, in2->loss);

		//in1->loss.vec() += loss.vec() * in2->val.vec();
		//in2->loss.vec() += loss.vec() * in1->val.vec();
		}
		 */

	public:
		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			return Node::typeEqual(other);
		}

		inline PExecute generate(bool bTrain);
};


class PMultiExecute :public Execute {
	public:
		vector<LDG::PTensor> vec_x1, vec_x2;
		vector<LDG::PTensor> vec_loss;
		vector<LDG::PTensor> vec_lx1, vec_lx2;
		vector<LDG::PTensor> vec_val;
		vector<LDG::PTensor> vec_in1_loss, vec_in2_loss;
		int sumDim;
		bool bTrain;

	public:
		inline void  forward() {
			int count = batch.size();
			sumDim = 0;
			for (int idx = 0; idx < count; idx++) {
				int dim0 = batch[idx]->val.shape().dims()[0];
				sumDim += dim0;
			}

			//y.init(sumDim);
			//x1.init(sumDim);
			//x2.init(sumDim);

			PMultiNode* ptr = (PMultiNode*)batch[0];
			drop_value = ptr->drop_value;
			vec_x1.resize(count);
			vec_x2.resize(count);
			vec_val.resize(count);
			for (int idx = 0; idx < count; idx++) {
				PMultiNode* ptr = (PMultiNode*)batch[idx];
				vec_x1[idx] = (&ptr->in1->val);
				vec_x2[idx] = (&ptr->in2->val);
				vec_val[idx] = (&ptr->val);
				ptr->degree = -1;
			}
			/*
			   int offset = 0;
			   for (int idx = 0; idx < count; idx++) {
			   PMultiNode* ptr = (PMultiNode*)batch[idx];
			   for (int idy = 0; idy < ptr->dim; idy++) {
			   x1[offset + idy] = ptr->in1->val[idy];
			   x2[offset + idy] = ptr->in2->val[idy];
			   }
			   offset += ptr->dim;
			   }
			 */

			//y.vec() = x1.vec() * x2.vec();

			DEV->Fmultiply(vec_x1, vec_x2, vec_val);

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
			/*
			   for (int idx = 0; idx < count; idx++) {
			   PMultiNode* ptr = (PMultiNode*)batch[idx];
			   ptr->forward_drop(bTrain);
			   }
			 */
			/*
			   offset = 0;
			   for (int idx = 0; idx < count; idx++) {
			   PMultiNode* ptr = (PMultiNode*)batch[idx];
			   for (int idy = 0; idy < ptr->dim; idy++) {
			   ptr->val[idy] = y[offset + idy];
			   }
			   offset += ptr->dim;
			   ptr->forward_drop(bTrain);
			   }
			 */
		}

		inline void  backward() {
			//Tensor1D ly, lx1, lx2;
			//ly.init(sumDim);
			//lx1.init(sumDim);
			//lx2.init(sumDim);

			int count = batch.size();
			vec_loss.resize(count);
			vec_lx1.resize(count);
			vec_lx2.resize(count);
			for (int idx = 0; idx < count; idx++) {
				PMultiNode* ptr = (PMultiNode*)batch[idx];
				vec_loss[idx] = (&ptr->loss);
				vec_lx1[idx] = (&ptr->lx1);
				vec_lx2[idx] = (&ptr->lx2);
			}

			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}

			/*
			   int offset = 0;
			   for (int idx = 0; idx < count; idx++) {
			   PMultiNode* ptr = (PMultiNode*)batch[idx];
			   ptr->backward_drop();
			   for (int idy = 0; idy < ptr->dim; idy++) {
			   ly[offset + idy] = ptr->loss[idy];
			   }
			   offset += ptr->dim;
			   }
			 */

			//lx1.vec() = ly.vec() * x2.vec();
			//lx2.vec() = ly.vec() * x1.vec();

			DEV->Fmultiply(vec_loss, vec_x2, vec_lx1);
			DEV->Fmultiply(vec_loss, vec_x1, vec_lx2);

			vec_in1_loss.resize(count);
			vec_in2_loss.resize(count);
			for (int idx = 0; idx < count; idx++) {
				PMultiNode* ptr = (PMultiNode*)batch[idx];
				vec_in1_loss[idx] = (&ptr->in1->loss);
				vec_in2_loss[idx] = (&ptr->in2->loss);
			}
			DEV->Fadd_inplace(vec_in1_loss, vec_lx1);
			DEV->Fadd_inplace(vec_in2_loss, vec_lx2);

			/*
			   offset = 0;
			   for (int idx = 0; idx < count; idx++) {
			   PMultiNode* ptr = (PMultiNode*)batch[idx];
			   for (int idy = 0; idy < ptr->dim; idy++) {
			   ptr->in1->loss[idy] += lx1[offset + idy];
			   ptr->in2->loss[idy] += lx2[offset + idy];
			   }
			   offset += ptr->dim;
			   }
			 */
		}

};

inline PExecute PMultiNode::generate(bool bTrain) {
	PMultiExecute* exec = new PMultiExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}
#endif
