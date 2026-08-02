// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

alignas(64) constexpr std::uint8_t INTERLEAVE_FIRST[64] = {
    0,64,1,65,2,66,3,67,4,68,5,69,6,70,7,71,
    8,72,9,73,10,74,11,75,12,76,13,77,14,78,15,79,
    16,80,17,81,18,82,19,83,20,84,21,85,22,86,23,87,
    24,88,25,89,26,90,27,91,28,92,29,93,30,94,31,95,
};

alignas(64) constexpr std::uint8_t INTERLEAVE_SECOND[64] = {
    32,96,33,97,34,98,35,99,36,100,37,101,38,102,39,103,
    40,104,41,105,42,106,43,107,44,108,45,109,46,110,47,111,
    48,112,49,113,50,114,51,115,52,116,53,117,54,118,55,119,
    56,120,57,121,58,122,59,123,60,124,61,125,62,126,63,127,
};

inline std::int32_t reduce_i32(__m512i value) {
    return _mm512_reduce_add_epi32(value);
}

inline float dot_signed_bytes_f32(__m512i weights, const float* inputs) {
    __m512 products = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_castsi512_si128(weights))),
        _mm512_loadu_ps(inputs)
    );
    products = _mm512_fmadd_ps(
        _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(weights, 1))
        ),
        _mm512_loadu_ps(inputs + 16),
        products
    );
    products = _mm512_fmadd_ps(
        _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(weights, 2))
        ),
        _mm512_loadu_ps(inputs + 32),
        products
    );
    products = _mm512_fmadd_ps(
        _mm512_cvtepi32_ps(
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(weights, 3))
        ),
        _mm512_loadu_ps(inputs + 48),
        products
    );
    return _mm512_reduce_add_ps(products);
}

}  // namespace

extern "C" __declspec(dllexport) int tokenql_q4_gemm_f32(
    const float* inputs,
    int tokens,
    const std::uint8_t* packed,
    const float* weight_scales,
    int rows,
    int columns,
    int block_size,
    float* output,
    int threads
) {
    if (!inputs || !packed || !weight_scales || !output || tokens < 1 ||
        rows < 1 || columns < 1 || block_size != 128 || columns % 128 != 0) {
        return -1;
    }

    const int blocks = columns / 128;
    const int packed_columns = columns / 2;
    const int worker_count = std::max(1, threads);
    const __m512i nibble_mask = _mm512_set1_epi8(0x0f);
    const __m512i sign_bias = _mm512_set1_epi8(0x08);
    const __m512i first_indices = _mm512_load_si512(INTERLEAVE_FIRST);
    const __m512i second_indices = _mm512_load_si512(INTERLEAVE_SECOND);
    const long long tasks = static_cast<long long>(tokens) * rows;

#pragma omp parallel for num_threads(worker_count) schedule(static)
    for (long long task = 0; task < tasks; ++task) {
        const int token = static_cast<int>(task / rows);
        const int row = static_cast<int>(task - static_cast<long long>(token) * rows);
        const float* activation = inputs + static_cast<std::size_t>(token) * columns;
        const std::uint8_t* weight = packed + static_cast<std::size_t>(row) * packed_columns;
        const float* scales = weight_scales + static_cast<std::size_t>(row) * blocks;
        float result = 0.0f;

        for (int block = 0; block < blocks; ++block) {
            const __m512i nibbles = _mm512_loadu_si512(weight + block * 64);
            const __m512i low = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(nibbles, nibble_mask), sign_bias
                ),
                sign_bias
            );
            const __m512i high = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(
                        _mm512_srli_epi16(nibbles, 4), nibble_mask
                    ),
                    sign_bias
                ),
                sign_bias
            );
            const __m512i weights_first = _mm512_permutex2var_epi8(
                low, first_indices, high
            );
            const __m512i weights_second = _mm512_permutex2var_epi8(
                low, second_indices, high
            );
            const float dot =
                dot_signed_bytes_f32(weights_first, activation + block * 128) +
                dot_signed_bytes_f32(weights_second, activation + block * 128 + 64);
            result += dot * scales[block];
        }
        output[static_cast<std::size_t>(token) * rows + row] = result;
    }
    return 0;
}

extern "C" __declspec(dllexport) int tokenql_q4_gemm_f32_pair(
    const float* inputs,
    int tokens,
    const std::uint8_t* packed_first,
    const float* scales_first,
    const std::uint8_t* packed_second,
    const float* scales_second,
    int rows,
    int columns,
    int block_size,
    float* output_first,
    float* output_second,
    int threads
) {
    if (!inputs || !packed_first || !scales_first || !packed_second ||
        !scales_second || !output_first || !output_second || tokens < 1 ||
        rows < 1 || columns < 1 || block_size != 128 || columns % 128 != 0) {
        return -1;
    }

    const int blocks = columns / 128;
    const int packed_columns = columns / 2;
    const int worker_count = std::max(1, threads);
    const __m512i nibble_mask = _mm512_set1_epi8(0x0f);
    const __m512i sign_bias = _mm512_set1_epi8(0x08);
    const __m512i first_indices = _mm512_load_si512(INTERLEAVE_FIRST);
    const __m512i second_indices = _mm512_load_si512(INTERLEAVE_SECOND);
    const long long tasks = static_cast<long long>(tokens) * rows;

#pragma omp parallel for num_threads(worker_count) schedule(static)
    for (long long task = 0; task < tasks; ++task) {
        const int token = static_cast<int>(task / rows);
        const int row = static_cast<int>(
            task - static_cast<long long>(token) * rows
        );
        const float* activation =
            inputs + static_cast<std::size_t>(token) * columns;
        const std::uint8_t* weight_first =
            packed_first + static_cast<std::size_t>(row) * packed_columns;
        const std::uint8_t* weight_second =
            packed_second + static_cast<std::size_t>(row) * packed_columns;
        const float* row_scales_first =
            scales_first + static_cast<std::size_t>(row) * blocks;
        const float* row_scales_second =
            scales_second + static_cast<std::size_t>(row) * blocks;
        float result_first = 0.0f;
        float result_second = 0.0f;

        for (int block = 0; block < blocks; ++block) {
            const __m512i nibbles_first =
                _mm512_loadu_si512(weight_first + block * 64);
            const __m512i nibbles_second =
                _mm512_loadu_si512(weight_second + block * 64);
            const __m512i low_first = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(nibbles_first, nibble_mask), sign_bias
                ),
                sign_bias
            );
            const __m512i high_first = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(
                        _mm512_srli_epi16(nibbles_first, 4), nibble_mask
                    ),
                    sign_bias
                ),
                sign_bias
            );
            const __m512i low_second = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(nibbles_second, nibble_mask), sign_bias
                ),
                sign_bias
            );
            const __m512i high_second = _mm512_sub_epi8(
                _mm512_xor_si512(
                    _mm512_and_si512(
                        _mm512_srli_epi16(nibbles_second, 4), nibble_mask
                    ),
                    sign_bias
                ),
                sign_bias
            );
            const __m512i weights_first_low = _mm512_permutex2var_epi8(
                low_first, first_indices, high_first
            );
            const __m512i weights_first_high = _mm512_permutex2var_epi8(
                low_first, second_indices, high_first
            );
            const __m512i weights_second_low = _mm512_permutex2var_epi8(
                low_second, first_indices, high_second
            );
            const __m512i weights_second_high = _mm512_permutex2var_epi8(
                low_second, second_indices, high_second
            );
            const float* block_inputs = activation + block * 128;
            result_first += (
                dot_signed_bytes_f32(weights_first_low, block_inputs) +
                dot_signed_bytes_f32(weights_first_high, block_inputs + 64)
            ) * row_scales_first[block];
            result_second += (
                dot_signed_bytes_f32(weights_second_low, block_inputs) +
                dot_signed_bytes_f32(weights_second_high, block_inputs + 64)
            ) * row_scales_second[block];
        }
        const std::size_t output_index =
            static_cast<std::size_t>(token) * rows + row;
        output_first[output_index] = result_first;
        output_second[output_index] = result_second;
    }
    return 0;
}

extern "C" __declspec(dllexport) int tokenql_q4_gemm(
    const float* inputs,
    int tokens,
    const std::uint8_t* packed,
    const float* weight_scales,
    int rows,
    int columns,
    int block_size,
    float* output,
    int threads
) {
    if (!inputs || !packed || !weight_scales || !output || tokens < 1 ||
        rows < 1 || columns < 1 || block_size != 128 || columns % 128 != 0) {
        return -1;
    }

    try {
        const int blocks = columns / 128;
        const int packed_columns = columns / 2;
        std::vector<std::int8_t> quantized(
            static_cast<std::size_t>(tokens) * columns
        );
        std::vector<float> input_scales(
            static_cast<std::size_t>(tokens) * blocks
        );
        std::vector<std::int32_t> input_sums(
            static_cast<std::size_t>(tokens) * blocks
        );

        const int worker_count = std::max(1, threads);
#pragma omp parallel for num_threads(worker_count) schedule(static)
        for (int task = 0; task < tokens * blocks; ++task) {
            const int token = task / blocks;
            const int block = task - token * blocks;
            const float* source = inputs + static_cast<std::size_t>(token) * columns + block * 128;
            std::int8_t* target = quantized.data() + static_cast<std::size_t>(token) * columns + block * 128;
            float maximum = 0.0f;
            for (int index = 0; index < 128; ++index) {
                maximum = std::max(maximum, std::fabs(source[index]));
            }
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            const float inverse = 1.0f / scale;
            std::int32_t sum = 0;
            for (int index = 0; index < 128; ++index) {
                int value = static_cast<int>(std::nearbyint(source[index] * inverse));
                value = std::max(-127, std::min(127, value));
                target[index] = static_cast<std::int8_t>(value);
                sum += value;
            }
            input_scales[task] = scale;
            input_sums[task] = sum;
        }

        const __m512i nibble_mask = _mm512_set1_epi8(0x0f);
        const __m512i sign_bias = _mm512_set1_epi8(0x08);
        const __m512i first_indices = _mm512_load_si512(INTERLEAVE_FIRST);
        const __m512i second_indices = _mm512_load_si512(INTERLEAVE_SECOND);
        const long long tasks = static_cast<long long>(tokens) * rows;

#pragma omp parallel for num_threads(worker_count) schedule(static)
        for (long long task = 0; task < tasks; ++task) {
            const int token = static_cast<int>(task / rows);
            const int row = static_cast<int>(task - static_cast<long long>(token) * rows);
            const std::int8_t* activation = quantized.data() + static_cast<std::size_t>(token) * columns;
            const std::uint8_t* weight = packed + static_cast<std::size_t>(row) * packed_columns;
            const float* scales = weight_scales + static_cast<std::size_t>(row) * blocks;
            const float* activation_scales = input_scales.data() + static_cast<std::size_t>(token) * blocks;
            const std::int32_t* activation_sums = input_sums.data() + static_cast<std::size_t>(token) * blocks;
            float result = 0.0f;

            for (int block = 0; block < blocks; ++block) {
                const __m512i nibbles = _mm512_loadu_si512(weight + block * 64);
                // The portable Q4 files store signed two's-complement
                // nibbles. XORing their sign bit converts them to the biased
                // unsigned [0, 15] form required by VPDPBUSD; subtracting
                // 8*sum(input) below restores the original signed dot.
                const __m512i low = _mm512_xor_si512(
                    _mm512_and_si512(nibbles, nibble_mask), sign_bias
                );
                const __m512i high = _mm512_xor_si512(
                    _mm512_and_si512(
                        _mm512_srli_epi16(nibbles, 4), nibble_mask
                    ),
                    sign_bias
                );
                const __m512i weights_first = _mm512_permutex2var_epi8(
                    low, first_indices, high
                );
                const __m512i weights_second = _mm512_permutex2var_epi8(
                    low, second_indices, high
                );
                const __m512i inputs_first = _mm512_loadu_si512(activation + block * 128);
                const __m512i inputs_second = _mm512_loadu_si512(activation + block * 128 + 64);
                __m512i products = _mm512_dpbusd_epi32(
                    _mm512_setzero_si512(), weights_first, inputs_first
                );
                products = _mm512_dpbusd_epi32(
                    products, weights_second, inputs_second
                );
                const std::int32_t dot = reduce_i32(products) - 8 * activation_sums[block];
                result += static_cast<float>(dot) * scales[block] * activation_scales[block];
            }
            output[static_cast<std::size_t>(token) * rows + row] = result;
        }
        return 0;
    } catch (...) {
        return -2;
    }
}
