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

#ifndef __GGMA_TRANSPOSE_H__
#define __GGMA_TRANSPOSE_H__

#include <cstddef>
#include <cstdint>

namespace ggma
{

/**
 * @brief Apply a 4D axis permutation to a flat byte buffer in-place.
 *
 * @param data          Pointer to the raw tensor data (modified in-place)
 * @param perm          Permutation string of length 4, e.g. "0213".
 *                      perm[i] = j means output axis i reads from input axis j.
 * @param dims          Array of 4 dimension sizes [d0, d1, d2, d3]
 * @param element_bytes Size of one element in bytes (e.g. 4 for float32)
 */
void transpose_4d(uint8_t *data, const char *perm, const size_t dims[4], size_t element_bytes);

} // namespace ggma

#endif // __GGMA_TRANSPOSE_H__