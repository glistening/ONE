/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__

#include "../ExternalContext.h"
#include "../Tensor.h"

#include <exec/train/ITrainableFunction.h>
#include <ops/FullyConnectedLayer.h>

namespace onert::backend::train::ops
{

class FullyConnectedLayer : public exec::train::ITrainableFunction,
                            public cpu::ops::FullyConnectedLayer
{
public:
  FullyConnectedLayer();
  ~FullyConnectedLayer();

public:
  void configureBackward(const IPortableTensor *input, const IPortableTensor *weights,
                         IPortableTensor *output, IPortableTensor *back_prop_input,
                         IPortableTensor *grad_weights, IPortableTensor *grad_bias,
                         const IPortableTensor *back_prop_output, ir::Activation activation,
                         ir::FullyConnectedWeightsFormat weights_format);

  void forward(bool training) override;
  void backward() override;

private:
  void backwardFloat32();

private:
  IPortableTensor *_grad_weights;
  IPortableTensor *_grad_bias;
  IPortableTensor *_back_prop_input;
  const IPortableTensor *_back_prop_output;

  // TODO Optimize memory
  std::unique_ptr<Tensor> _transposed_weights;
  std::unique_ptr<Tensor> _transposed_input;
  std::unique_ptr<Tensor> _transposed_back_prop_output;
  std::unique_ptr<Tensor> _act_back_prop_output;
};

} // namespace onert::backend::train::ops

#endif // __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__
