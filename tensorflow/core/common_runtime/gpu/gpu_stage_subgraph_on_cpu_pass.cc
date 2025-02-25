/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class StageSubGraphOnCPUPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr) {
      return Status::OK();
    }
    
    bool is_enable_stage_subgraph_on_cpu =
        options.session_options->config.graph_options()
            .optimizer_options().stage_subgraph_on_cpu();
    if (is_enable_stage_subgraph_on_cpu) {
      LOG(INFO) << "Run StageSubGraphOnCPU Optimization";
    } else {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");

    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());
    
    // Get CPU Device
    std::string cpu_device_name="";
    const DeviceSet* device_set = options.device_set;
    GetCPUDevice(cpu_device_name, device_set);
    if (cpu_device_name.empty()) {
      LOG(INFO) << "Failed to Get CPU Device. "
		<< "StageSubGraphOnCPU Optimization is disabled.";
	return Status::OK();
    }

    // Place Stage SubGraph on CPU.
    PlaceStageSubGraphOnCPU(cpu_device_name, new_graph.get());
      
    options.graph->swap(new_graph);
    return Status::OK();
  }

 private:
  void GetCPUDevice(std::string& cpu_device_name, const DeviceSet* device_set) {
    const auto& devices = device_set->devices();
    for (auto iter = devices.begin(); iter != devices.end(); iter++) {
      if ((*iter)->device_type() == "CPU") {
	cpu_device_name = (*iter)->name();
	return;
      }
    }
  }

  void PlaceStageSubGraphOnCPU(const std::string& cpu_device_name,
			       Graph* graph) {    
    for (Node* n : graph->op_nodes()) {
      if (n->IsStage()) {
	auto set_stage_subgraph_node_device = [cpu_device_name](Node *node) {
	  node->set_assigned_device_name(cpu_device_name);
	};
        ReverseDFSFrom(*graph, {n},
                       std::move(set_stage_subgraph_node_device), nullptr);        
      } else if (n->IsUnstage() ||
		 n->type_string() == "TensorBufferCancel" ||
		 n->type_string() == "TensorBufferClose") {
	n->set_assigned_device_name(cpu_device_name);
      }
    }
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 25,
                      StageSubGraphOnCPUPass);
  
} // End of namespace tensorflow

#endif // End of GOOGLE_CUDA    
