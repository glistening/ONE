/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Config.h"
#include "KVCache.h"
#include "Transpose.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace ggma
{

// Convert KVCacheDataType to string representation
const char *to_string(KVCacheDataType type)
{
  switch (type)
  {
    case KVCacheDataType::FLOAT32:
      return "FLOAT32";
    case KVCacheDataType::UINT8:
      return "UINT8";
    default:
      return "UNKNOWN";
  }
}

// Convert string to KVCacheDataType
KVCacheDataType from_string(const std::string &type_str)
{
  if (type_str == "FLOAT32" || type_str == "float32")
  {
    return KVCacheDataType::FLOAT32;
  }
  else if (type_str == "UINT8" || type_str == "uint8")
  {
    return KVCacheDataType::UINT8;
  }
  throw std::runtime_error("Unsupported KV cache data type: " + type_str);
}

// Check if KVCacheDataType is supported
bool is_supported_type(KVCacheDataType type)
{
  switch (type)
  {
    case KVCacheDataType::FLOAT32:
    case KVCacheDataType::UINT8:
      return true;
    default:
      return false;
  }
}

void KVCache::transpose(bool is_k_cache, const char *perm)
{
  std::vector<std::vector<uint8_t>> &cache_vector = is_k_cache ? k : v;
  const size_t element_bytes = element_size();

  for (size_t i = 0; i < cache_vector.size(); ++i)
  {
    transpose_4d(cache_vector[i].data(), perm, _shape, element_bytes);
  }

  // Update _shape to reflect the permuted layout.
  size_t old[4] = {_shape[0], _shape[1], _shape[2], _shape[3]};
  for (int i = 0; i < 4; i++)
    _shape[i] = old[perm[i] - '0'];
}

void KVCache::init(const ggma::GGMAConfig &cfg, int max_total_tokens)
{
  if (cfg.model.n_layers <= 0)
    throw std::runtime_error("n_layers not properly initialized");

  // Set KV cache data type from config
  data_type = cfg.kv_cache_type;

  // Store the logical shape of each layer's buffer: [1, seq_len, num_heads, head_dim]
  _shape[0] = 1;
  _shape[1] = max_total_tokens;
  _shape[2] = cfg.model.num_attention_heads;
  _shape[3] = cfg.model.hidden_size / cfg.model.num_attention_heads;

  // Allocate space for K and V caches for each layer
  // Total: n_layers * 2 vectors (K and V for each layer)
  k.resize(cfg.model.n_layers);
  v.resize(cfg.model.n_layers);

  for (int i = 0; i < cfg.model.n_layers; ++i)
  {
    // buffer_size = prod(_shape) * element_size
    size_t buffer_size = _shape[0] * _shape[1] * _shape[2] * _shape[3] * element_size();
    k[i].resize(buffer_size, 0);
    v[i].resize(buffer_size, 0);
  }
}

} // namespace ggma
