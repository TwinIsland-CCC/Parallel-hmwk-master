#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <sstream>
#include <unordered_map>

using namespace std;

class Process {
public:
    Process() : id_(0), parent_(nullptr), v_(1.0) {}
    Process(int id, Process* parent) : id_(id), parent_(parent), v_(0.0) {}

    // 分配任务
    Process* alloc(int id) {
        auto child = new Process(id, this);
        children_.push_back(child);
        
        double new_v = 2.0 * v_;
        v_ = new_v;
        child->v_ = new_v;
        
        return child;
    }
    
    void terminate() {
        if (!children_.empty()) {
            cout << "only leaf node can do this" << endl;
            return;
        }
        if (parent_) {
            parent_->v_ = (double)(parent_->v_ * v_) / (double)(parent_->v_ + v_);
            
            auto& siblings = parent_->children_;
            siblings.erase(remove(siblings.begin(), siblings.end(), this), siblings.end());
        }
    }
    
    bool is_terminated() const {
        return abs(v_ - 1.0) < 1e-9;
    }
    
    double get_v_() const {
        return v_;
    }

    // 获取原始权重
    double get_v_original() const {
        return 1.0 / v_;
    }

    int get_id_() const {
        return id_;
    } 

private:
    int id_;  // 编号
    Process* parent_;                
    double v_;  // 倒数
    vector<Process*> children_; 
};

int main(int argc, char** argv) {
    Process root;
    string line;

    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "cannot open file: " << argv[1] << endl;
        return 1;
    }

    unordered_map<int, Process*> p_set;
    while (getline(infile, line)) {
        cout << line << endl;
        istringstream iss(line);
        string cmd;
        int from, to;
        iss >> cmd;

        if (cmd == "start") {
            int root_id;
            iss >> root_id;
            p_set[root_id] = &root;
        } else if (cmd == "send") {
            iss >> from >> to;
            auto ptr = p_set[from]->alloc(to);
            p_set[to] = ptr;
        } else if (cmd == "done") {
            int node;
            iss >> node;
            p_set[node]->terminate();
        }
    }
    cout << root.is_terminated() << endl;
    cout << root.get_v_() << endl;
    return 0;
}