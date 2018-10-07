#ifndef POOLING
#define POOLING

/*
*  Pooling.h:
*  pool operation, max, min, average and sum pooling
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"


class PoolNode : public Node {
  public:
    vector<PNode> ins;

  public:
    PoolNode() : Node() {
        ins.clear();
    }

    ~PoolNode() {
        ins.clear();
    }

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
    }

    inline void setParam(int maxsize) {
    }


    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for max|min|sum|avg pooling" << std::endl;
            return;
        }
        int nSize = x.size();
        ins.clear();
        for (int i = 0; i < nSize; i++) {
			int val_dim0 = x[i]->val.shape()[0];
            if (val_dim0 != dim) {
                std::cout << "input matrixes are not matched" << std::endl;
                clearValue();
                return;
            }
            ins.push_back(x[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
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

    virtual void compute() = 0;

	virtual void backward() = 0;
};

class MaxPoolNode : public PoolNode {
  public:
	IndexPtr index;
    MaxPoolNode() : PoolNode() {
        node_type = "max-pooling";
    }

    void init(int ndim, dtype dropout) {
		Node::init(ndim, dropout);
		DEV->init_index_ptr(index, ndim);
	}
	~MaxPoolNode(){
	}

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
    inline void compute() {
        int nSize = ins.size();
		//LDG::Tensor in_x;
		//DEV->init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		//DEV->concat(vec_ins, in_x);
		DEV->FMaxPooling(vec_ins, val, index.get_ptr());
    }

    inline void backward() {
        int nSize = ins.size();

		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			vec_in_loss.push_back(&ins[i]->loss);
		}

		DEV->DMaxPooling(loss, vec_in_loss, index.get_ptr());

    }

};

class AvgPoolNode : public PoolNode {
	public:
		AvgPoolNode() : PoolNode() {
			node_type = "avg-pooling";
		}

	public:
		//Be careful that the row is the dim of input vector, and the col is the number of input vectors
		//Another point is that we change the input vectors directly.
		inline void compute() {
			int nSize = ins.size();

			vector<LDG::PTensor> vec_ins;
			for (int i = 0; i < nSize; ++i) {
				vec_ins.push_back(&ins[i]->val);
			}

			DEV->FAvgPooling(vec_ins, val);
		}

		inline void backward() {
			int nSize = ins.size();
			vector<LDG::PTensor> vec_ins_loss;
			for (int i = 0; i < nSize; i++) {
				vec_ins_loss.push_back(&ins[i]->loss);
			}
			DEV->DAvgPooling(loss, vec_ins_loss);
		}
};

class SumPoolNode : public PoolNode {
  public:
    SumPoolNode() : PoolNode() {
        node_type = "sum-pooling";
    }

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
    inline void compute() {
        int nSize = ins.size();
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		DEV->FSumPooling(vec_ins, val);
    }

    inline void backward() {
        int nSize = ins.size();
		vector<LDG::PTensor> vec_ins_loss;
        for (int i = 0; i < nSize; i++) {
			vec_ins_loss.push_back(&ins[i]->loss);
        }
		DEV->DSumPooling(loss, vec_ins_loss);
    }
};

class MinPoolNode : public PoolNode {
  public:
	IndexPtr index;

    MinPoolNode() : PoolNode() {
        node_type = "min-pooling";
    }

    void init(int ndim, dtype dropout) {
		Node::init(ndim, dropout);
		DEV->init_index_ptr(index, ndim);
	}
	~MinPoolNode(){
	}

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.

    inline void compute() {
        int nSize = ins.size();

		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}

		DEV->FMinPooling(vec_ins, val, index.get_ptr());
    }

    inline void backward() {
        int nSize = ins.size();

		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			vec_in_loss.push_back(&ins[i]->loss);
		}

		DEV->DMinPooling(loss, vec_in_loss, index.get_ptr());

    }
};


//#if USE_GPU
//class PoolExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PoolNode* ptr = (PoolNode*)batch[idx];
//      ptr->compute();
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PoolNode* ptr = (PoolNode*)batch[idx];
//      ptr->backward_drop();
//      ptr->backward();
//    }
//  }
//};
//
//inline PExecute PoolNode::generate(bool bTrain) {
//  PoolExecute* exec = new PoolExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else

class PoolExecute : public Execute {
  public:
    bool bTrain;
	vector<LDG::PTensor> vec_val;
	vector<LDG::PTensor> vec_loss;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
		PoolNode* ptr = (PoolNode*)batch[0];
		drop_value = ptr->drop_value;
		vec_val.resize(count);
        for (int idx = 0; idx < count; idx++) {
            PoolNode* ptr = (PoolNode*)batch[idx];
			vec_val[idx] = (&ptr->val);
            ptr->compute();
			ptr->degree = -1;
            //ptr->forward_drop(bTrain);
        }

		if(drop_value > 0) {
			if(bTrain)
				DEV->Fdropout(vec_val, drop_value, mask, vec_val);
			else
				DEV->Fdropout(vec_val, drop_value, vec_val);
		}
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
		vec_loss.resize(count);
        for (int idx = 0; idx < count; idx++) {
            PoolNode* ptr = (PoolNode*)batch[idx];
			vec_loss[idx] = (&ptr->loss);
		}
		if (drop_value > 0) {
			DEV->Ddropout(vec_loss, mask);
		}
        for (int idx = 0; idx < count; idx++) {
            PoolNode* ptr = (PoolNode*)batch[idx];
            //ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute PoolNode::generate(bool bTrain) {
    PoolExecute* exec = new PoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
//#endif

#endif
