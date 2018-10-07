#ifndef PAddOP
#define PAddOP

/*
 *  PAddOP.h:
 *  (pointwise) add
 *
 *  Created on: June 13, 2017
 *      Author: mszhang
 */

//#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PAddNode : public Node {
	public:
		vector<PNode> ins;
		vector<LDG::PTensor> vec_in_val;
		vector<LDG::PTensor> vec_ins_loss;
		vector<LDG::PTensor> vec_loss;

		~PAddNode() {
			ins.clear();
		}
	public:
		PAddNode() : Node() {
			ins.clear();
			node_type = "point-add";
		}

		inline void clearValue() {
			ins.clear();
			Node::clearValue();
		}

	public:
		void forward(Graph *cg, const vector<PNode>& x) {
			if (x.size() == 0) {
				std::cout << "empty inputs for add" << std::endl;
				return;
			}

			ins.clear();
			for (int i = 0; i < x.size(); i++) {
				int dim0 = x[i]->val.shape().dims()[0];
				if (dim0 == dim) {
					ins.push_back(x[i]);
				} else {
					std::cout << "dim does not match" << std::endl;
				}
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
			if (x1->dim == dim) {
				ins.push_back(x1);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x2->dim == dim) {
				ins.push_back(x2);
			} else {
				std::cout << "dim does not match" << std::endl;
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
			ins.clear();
			if (x1->dim == dim) {
				ins.push_back(x1);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x2->dim == dim) {
				ins.push_back(x2);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x3->dim == dim) {
				ins.push_back(x3);
			} else {
				std::cout << "dim does not match" << std::endl;
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
			ins.clear();
			if (x1->dim == dim) {
				ins.push_back(x1);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x2->dim == dim) {
				ins.push_back(x2);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x3->dim == dim) {
				ins.push_back(x3);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x4->dim == dim) {
				ins.push_back(x4);
			} else {
				std::cout << "dim does not match" << std::endl;
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
			ins.clear();
			if (x1->dim == dim) {
				ins.push_back(x1);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x2->dim == dim) {
				ins.push_back(x2);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x3->dim == dim) {
				ins.push_back(x3);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x4->dim == dim) {
				ins.push_back(x4);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x5->dim == dim) {
				ins.push_back(x5);
			} else {
				std::cout << "dim does not match" << std::endl;
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

		void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
			ins.clear();
			if (x1->dim == dim) {
				ins.push_back(x1);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x2->dim == dim) {
				ins.push_back(x2);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x3->dim == dim) {
				ins.push_back(x3);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x4->dim == dim) {
				ins.push_back(x4);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x5->dim == dim) {
				ins.push_back(x5);
			} else {
				std::cout << "dim does not match" << std::endl;
			}
			if (x6->dim == dim) {
				ins.push_back(x6);
			} else {
				std::cout << "dim does not match" << std::endl;
			}

			degree = 0;
			int nSize = ins.size();
			for (int i = 0; i < nSize; ++i) {
				ins[i]->addParent(this);
			}

			cg->addNode(this);
		}

	public:
		/*
		   inline void compute() {
		   int nSize = ins.size();
		   DEV->zero(val);
	//val.zero();
	vec_in_val.resize(nSize);
	for (int i = 0; i < nSize; ++i) {
	vec_in_val[i] = (&ins[i]->val);
	//DEV->Fadd(val, ins[i]->val, val);
	//for (int idx = 0; idx < dim; idx++) {
	//val[idx] += ins[i]->val[idx];
	//}
	}
	DEV->Fadd_inplace(val, vec_in_val);
	}


	void backward() {
	int nSize = ins.size();
	vec_ins_loss.resize(nSize);
	vec_loss.resize(nSize);
	for (int i = 0; i < nSize; ++i) {
	vec_ins_loss[i] = (&ins[i]->loss);
	vec_loss[i] = (&loss);
	//DEV->Fadd(ins[i]->loss, loss, ins[i]->loss);
	//for (int idx = 0; idx < dim; idx++) {
	//ins[i]->loss[idx] += loss[idx];
	//}
	}
	DEV->Fadd_inplace(vec_ins_loss, vec_loss);
	}
		 */


	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			return Node::typeEqual(other);
		}

};

//#if USE_GPU
//class PAddExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//
//    for (int idx = 0; idx < count; idx++) {
//      PAddNode* ptr = (PAddNode*)batch[idx];
//      ptr->compute();
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PAddNode* ptr = (PAddNode*)batch[idx];
//      ptr->backward_drop();
//      ptr->backward();
//    }
//  }
//};
//
//
//inline PExecute PAddNode::generate(bool bTrain) {
//  PAddExecute* exec = new PAddExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else
class PAddExecute : public Execute {
	public:
		bool bTrain;
	public:
		inline void  forward() {
			int count = batch.size();
			//#pragma omp parallel for schedule(static,1)
			vector<vector<LDG::PTensor> > vec_vec_x;
			vector<LDG::PTensor> vec_val;
			PAddNode* ptr = (PAddNode*)batch[0];
			drop_value = ptr->drop_value;
			for (int idx = 0; idx < count; idx++) {
				PAddNode* ptr = (PAddNode*)batch[idx];
				vector<LDG::PTensor> vec_x;
				int n = ptr->ins.size();
				for(int idy = 0; idy < n; idy++)
					vec_x.push_back(&ptr->ins[idy]->val);
				vec_vec_x.push_back(vec_x);
				vec_val.push_back(&ptr->val);
				ptr->degree = -1;
				//ptr->compute();
			}

			DEV->Fadd(vec_vec_x, vec_val);

			if(drop_value > 0) {
				if(bTrain)
					DEV->Fdropout(vec_val, drop_value, mask, vec_val);
				else
					DEV->Fdropout(vec_val, drop_value, vec_val);
			}
			/*
			   for (int idx = 0; idx < count; idx++) {
			   PAddNode* ptr = (PAddNode*)batch[idx];
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
				PAddNode* ptr = (PAddNode*)batch[idx];
				vec_loss.push_back(&ptr->loss);
				int n = ptr->ins.size();
				vector<LDG::PTensor> vec_in_loss;
				for(int idy = 0; idy < n; idy++) {
					vec_in_loss.push_back(&ptr->ins[idy]->loss);
				}
				vec_vec_in_loss.push_back(vec_in_loss);
				//ptr->backward_drop();
				//ptr->backward();
			}
			if (drop_value > 0) {
				DEV->Ddropout(vec_loss, mask);
			}
			DEV->Dadd(vec_vec_in_loss, vec_loss);
		}
};


inline PExecute PAddNode::generate(bool bTrain) {
	PAddExecute* exec = new PAddExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}
//#endif


#endif
