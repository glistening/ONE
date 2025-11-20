#include "ggml.h"
#include <cstdint>

extern "C" {

void circle_quantize_q4_0(const float* src, void* dst, int64_t n) {
    ggml_quantize_chunk(GGML_TYPE_Q4_0, src, dst, 0, 1, n, nullptr);
}

}
