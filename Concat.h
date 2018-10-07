#ifndef CONCAT
#define CONCAT

/*
 *  Concat.h:
 *  concatenatation operation.
 *
 *  Created on: Apr 22, 2017
 *      Author: mszhang
 */


#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class ConcatNode : public Node {
	public:
		vector<int> inDims;
		vector<PNode> ins;

	public:
		ConcatNode() : Node() {
			inDims.clear();
			ins.clear();
			node_type = "concat";
		}

		inline void clearValue() {
			Node::clearValue();
			inDims.clear();
			ins.clear();
		}

	public:
		void forward(Graph *cg, const vector<PNode>& x) {
			if (x.size() == 0) {
				std::cout << "empty inputs for concat" << std::endl;
				return;
			}

			ins.clear();
			for (int i = 0; i < x.size(); i++) {
				ins.push_back(x[i]);
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}


		void forward(Graph *cg, PNode x1, PNode x2) {
			ins.clear();
			ins.push_back(x1);
			ins.push_back(x2);

			degree = 0;
			for (int i = 0; i < 2; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
			ins.clear();
			ins.push_back(x1);
			ins.push_back(x2);
			ins.push_back(x3);

			degree = 0;
			for (int i = 0; i < 3; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
			ins.clear();
			ins.push_back(x1);
			ins.push_back(x2);
			ins.push_back(x3);
			ins.push_back(x4);

			degree = 0;
			for (int i = 0; i < 4; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
			ins.clear();
			ins.push_back(x1);
			ins.push_back(x2);
			ins.push_back(x3);
			ins.push_back(x4);
			ins.push_back(x5);

			degree = 0;
			for (int i = 0; i < 5; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
			ins.clear();
			ins.push_back(x1);
			ins.push_back(x2);
			ins.push_back(x3);
			ins.push_back(x4);
			ins.push_back(x5);
			ins.push_back(x6);

			degree = 0;
			for (int i = 0; i < 6; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}



	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			return Node::typeEqual(other);
		}

	public:
		/*
		   inline void compute() {
		   int nSize = ins.size();
		   inDims.clear();
		   int curDim = 0;
		   for (int i = 0; i < nSize; ++i) {
		   int in_dim = ins[i]->val.shape().dims()[0];
		   inDims.push_back(in_dim);
		   curDim += inDims[i];
		   }
		   if (curDim != dim) {
		   std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
		   return;
		   }

		   vector<LDG::PTensor> vec_ins;
		   for (int i = 0; i < nSize; ++i) {
		   vec_ins.push_back(&ins[i]->val);
		   }
		   DEV->concat(vec_ins, val);
		   }


		   void backward() {
		   int nSize = ins.size();
		   LDG::Tensor array_loss[nSize];
		   vector<LDG::PTensor> vec_loss, vec_in_loss;
		   for (int i = 0; i < nSize; ++i) {
		   DEV->init(array_loss[i], ins[i]->loss.shape());
		   vec_loss.push_back(&array_loss[i]);
		   vec_in_loss.push_back(&ins[i]->loss);
		   }
		   DEV->unconcat(loss, vec_loss);

		   DEV->Fadd_inplace(vec_in_loss, vec_loss);
		   }
		 */

};


//#if USE_GPU
//class ConcatExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      ConcatNode* ptr = (ConcatNode*)batch[idx];
//      ptr->compute();
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      ConcatNode* ptr = (ConcatNode*)batch[idx];
//      ptr->backward_drop();
//      ptr->backward();
//    }
//  }
//};
//
//inline PExecute ConcatNode::generate(bool bTrain) {
//  ConcatExecute* exec = new ConcatExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else
class ConcatExecute : public Execute {
	public:
		bool bTrain;
		vector<vector<LDG::PTensor> > vec_vec_x;
		vector<LDG::PTensor> vec_val;
	public:
		inline void  forward() {
			int count = batch.size();
			//#pragma omp parallel for schedule(static,1)
			vec_vec_x.clear();
			vec_val.clear();

			ConcatNode* ptr = (ConcatNode*)batch[0];
			drop_value = ptr->drop_value;

			for(int idx = 0; idx < count; idx++) {
				ConcatNode* ptr = (ConcatNode*)batch[idx];
				int ins_size = ptr->ins.size();
				vector<LDG::PTensor> vec_x;
				for(int idy = 0; idy < ins_size; idy++) {
					vec_x.push_back(&(ptr->ins[idy]->val));
				}
				vec_vec_x.push_back(vec_x);
				vec_val.push_back(&(ptr->val));
				ptr->degree = -1;
				//cout << ptr->val.shape().to_string() << endl;
				//ptr->compute();
			}
			//cout << "=====================" << endl;

			DEV->Fconcat(vec_vec_x, vec_val);

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}

			/*
			   for (int idx = 0; idx < count; idx++) {
			   ConcatNode* ptr = (ConcatNode*)batch[idx];
			//ptr->compute();
			ptr->forward_drop(bTrain);
			}
			 */
		}

		inline void backward() {
			int count = batch.size();
			//#pragma omp parallel for schedule(static,1)
			vector<LDG::PTensor> vec_loss;
			vector<vector<LDG::PTensor> > vec_vec_in_loss;
			for (int idx = 0; idx < count; idx++) {
				ConcatNode* ptr = (ConcatNode*)batch[idx];
				//ptr->backward_drop();
				//ptr->backward();
				vec_loss.push_back(&ptr->loss);
				int ins_size = ptr->ins.size();
				vector<LDG::PTensor> vec_in_loss;
				for(int idy = 0; idy < ins_size; idy++) {
					vec_in_loss.push_back(&(ptr->ins[idy]->loss));
				}
				vec_vec_in_loss.push_back(vec_in_loss);
			}
			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}
			DEV->Dconcat(vec_vec_in_loss, vec_loss);
		}
};

inline PExecute ConcatNode::generate(bool bTrain) {
	ConcatExecute* exec = new ConcatExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}
//#endif

#endif
