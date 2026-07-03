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

#include "Transpose.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ggma
{

void transpose_4d(uint8_t *data, const char *perm, const size_t dims[4], size_t element_bytes)
{
  if (perm == nullptr)
    throw std::runtime_error("transpose_4d: perm must not be null");

  if (strcmp(perm, "0213") != 0)
    throw std::runtime_error("transpose_4d: only \"0213\" permutation is supported");

  // "0213" swaps axis 1 and axis 2 while keeping axis 0 and axis 3 in place.
  // dims = [d0, d1, d2, d3]  ->  output is [d0, d2, d1, d3]
  const size_t d0 = dims[0];
  const size_t d1 = dims[1];
  const size_t d2 = dims[2];
  const size_t d3 = dims[3];
  // ggml-style byte strides for contiguous layout [d0, d1, d2, d3]
  const size_t nb1 = d3 * element_bytes;
  const size_t nb2 = d2 * nb1;
  const size_t nb3 = d1 * nb2;

  // Work on a temporary copy, then write back.
  std::vector<uint8_t> tmp(data, data + d0 * d1 * d2 * d3 * element_bytes);

  for (size_t a0 = 0; a0 < d0; ++a0)
  {
    for (size_t a1 = 0; a1 < d1; ++a1)
    {
      for (size_t a2 = 0; a2 < d2; ++a2)
      {
        // input [d0,d1,d2,d3]  ->  output [d0,d2,d1,d3]
        const uint8_t *src = tmp.data() + a0 * nb3 + a1 * nb2 + a2 * nb1;
        uint8_t *dst = data + a0 * nb3 + a2 * (d1 * nb1) + a1 * nb1;
        memcpy(dst, src, nb1);
      }
    }
  }
}

} // namespace ggma