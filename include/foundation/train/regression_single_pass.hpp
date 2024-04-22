#pragma once
#include "../../../external/threadpool/include/BS_thread_pool.hpp"
#include "foundation/graph/compiled_graph.hpp"
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;
namespace foundation {

namespace train {

void one_d_regression_train_single_pass(graph::CompiledGraph *compiled_graph, std::vector<torch::Tensor> &input,
                           std::vector<torch::Tensor> &target, int num_epochs, double learning_rate, bool evolving_loss = false, int early_stop_k = 15) {

  // Set the graph to training mode
  compiled_graph->train();

  if (compiled_graph->parameters().size() == 0) {
    return;
  }

  // optimizer
  torch::optim::Adam optimizer(compiled_graph->parameters(), learning_rate);

  // exponential decay of the loss for early stopping
  double current_expdecay_loss;
  std::vector<double> expdecay_losses;


  // Loop over the epochs
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Zero the gradients
    optimizer.zero_grad();

    // Forward pass
    auto output = compiled_graph->forward(input);
    torch::Tensor loss;

    double mseloss;
    if (!evolving_loss) {
      loss = torch::mse_loss(output[0], target[0]);
      mseloss = loss.item<double>();
    } else {

      auto losses = compiled_graph->forward_response(target);

      // loss is the average of all the losses
      loss = torch::mean(torch::stack(losses));

      {
        torch::NoGradGuard no_grad;
        mseloss = torch::mse_loss(output[0], target[0]).item<double>();
      }
    }


    // Backward pass
    loss.backward();

    // Update the weights
    optimizer.step();


    // exponential decay of the loss
    if (epoch == 0) {
      current_expdecay_loss = mseloss;
    } else {
      current_expdecay_loss = 0.9 * current_expdecay_loss + 0.1 * mseloss;
    }

    expdecay_losses.push_back(current_expdecay_loss);

    // early stopping
    if (epoch > early_stop_k) {
      if(expdecay_losses[epoch - early_stop_k] <= expdecay_losses[epoch]) {
        break;
      }
    }
  }
}

void one_d_regression_train_population(std::vector<graph::CompiledGraph *> &population, std::vector<torch::Tensor> &input,
                          std::vector<torch::Tensor> &target, int num_epochs, std::vector<double> learning_rates,
                          int num_threads = 12, bool evolving_loss = false, int early_stop_k = 15) {

  // thread pool
  auto pool = BS::thread_pool(num_threads);

  // run the training for each model in the population
  int idx = 0;
  for (auto model : population) {
    auto task = [&, model, num_epochs, lr = learning_rates[idx]]() {
      one_d_regression_train_single_pass(model, input, target, num_epochs, lr, evolving_loss, early_stop_k);
    };

    pool.detach_task(task);
    idx += 1;
  }

  // wait for all threads to finish
  pool.wait();
}


} // namespace train
} // namespace foundation
