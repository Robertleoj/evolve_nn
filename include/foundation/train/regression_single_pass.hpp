#pragma once
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "foundation/graph/compiled_graph.hpp"
#include "../../../external/threadpool/include/BS_thread_pool.hpp"

namespace py = pybind11;
namespace foundation {

namespace train {

    void train_mse_single_pass(
        graph::CompiledGraph* compiled_graph,
        std::vector<torch::Tensor>& input,
        std::vector<torch::Tensor>& target,
        int num_epochs,
        double learning_rate
    ) {

        // Set the graph to training mode
        compiled_graph->train();

        if(compiled_graph->parameters().size() == 0) {
            return;
        }

        // optimizer
        torch::optim::Adam optimizer(
            compiled_graph->parameters(), 
            learning_rate
        );

        // Loop over the epochs
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Zero the gradients
            optimizer.zero_grad();

            // Forward pass
            auto output = compiled_graph->forward(input);

            auto loss = torch::mse_loss(output[0], target[0]);

            // Backward pass
            loss.backward();

            // Update the weights
            optimizer.step();
        }
    }

    void train_population(
        std::vector<graph::CompiledGraph*>& population,
        std::vector<torch::Tensor>& input,
        std::vector<torch::Tensor>& target,
        int num_epochs,
        std::vector<double> learning_rates,
        int num_threads = 12
    ) {

        // thread pool
        auto pool = BS::thread_pool(num_threads);

        // run the training for each model in the population
        int idx = 0;
        for(auto model : population) {
            auto task = [&, model, num_epochs, lr=learning_rates[idx]]() {
                train_mse_single_pass(model, input, target, num_epochs, lr);
            };

            pool.detach_task(task);
            idx += 1;
        }

        // wait for all threads to finish
        pool.wait();

    }
}
}