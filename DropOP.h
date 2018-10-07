#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"


class DropNode :public Node {
	public:
		PNode in;

		dtype drop_value;
		bool bTrain;
	public:
		DropNode() : Node() {
			in = NULL;
			node_type = "dropout";
		}

		~DropNode() {
			in = NULL;
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;

		}

		void init(int ndim, dtype dropout) {
			Node::init(ndim, dropout);

			if (dropout > 0 && dropout <= 1) {
				drop_value = dropout;
			} else {
				drop_value = -1;
			}
		}

	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			return result;
		}

	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}
};

class DropExecute :public Execute {
	public:
		bool bTrain;
	public:
		inline void forward() {
			int count = batch.size();
			DropNode* ptr = (DropNode*)batch[0];

			drop_value = ptr->drop_value;
			vector<LDG::PTensor> vec_x, vec_val;
			for (int idx = 0; idx < count; idx++) {
				DropNode* ptr = (DropNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
				vec_val.push_back(&ptr->val);
				ptr->degree = -1;
			}

			if(bTrain)
				DEV->Fdropout(vec_x, drop_value, mask, vec_val);
			else
				DEV->Fdropout(vec_x, drop_value, vec_val);
		}

		inline void backward() {
			int count = batch.size();
			vector<LDG::PTensor> vec_loss, vec_in_loss;
			for (int idx = 0; idx < count; idx++) {
				DropNode* ptr = (DropNode*)batch[idx];
				vec_in_loss.push_back(&ptr->in->loss);
				vec_loss.push_back(&ptr->loss);
			}
			DEV->Ddropout(vec_in_loss, mask, vec_loss);
		}

};

inline PExecute DropNode::generate(bool bTrain) {
	DropExecute* exec = new DropExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}
