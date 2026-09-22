#include <stdio.h>
#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define INPUT_FILE "input.bin"
#define INPUT_MAGIC_SIZE 8
#define COMPUTE_ROUNDS 16

typedef struct {
    int32_t checksum;
    double elapsed_seconds;
} performance_ret;

typedef struct {
    int32_t* a;
    int32_t* b;
    size_t num_elems;
} dot_product_vec_args_t;

typedef int32_t (*DotProductVec)(int32_t* a, int32_t* b, size_t num_elems); 


int32_t dot_product_vec(int32_t* a, int32_t* b, size_t num_elems);
int32_t dot_product_vec_parallel(int32_t* a, int32_t* b, size_t num_elems);
performance_ret test_dot_product_vec(DotProductVec func, dot_product_vec_args_t args, size_t iterations);
static inline uint32x4_t dot_product_work_vec(int32x4_t vec_a, int32x4_t vec_b);
static inline uint32_t dot_product_work_scalar(int32_t a, int32_t b);
int read_exact(FILE* file, void* buffer, size_t num_bytes);
int read_u64_le(FILE* file, uint64_t* value);

int main() {

    FILE* file = fopen(INPUT_FILE, "rb");
    if (file == NULL) return 1;

    const size_t ITERATIONS = 100;
    const unsigned char expected_magic[INPUT_MAGIC_SIZE] = {'F', 'O', 'I', '3', '2', 'v', '1', '\0'};
    unsigned char magic[INPUT_MAGIC_SIZE] = {0};
    uint64_t block_count = 0;
    int status = 0;

    if (read_exact(file, magic, sizeof(magic)) != 0 || memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        fprintf(stderr, "invalid input file format\n");
        fclose(file);
        return 1;
    }

    if (read_u64_le(file, &block_count) != 0) {
        fprintf(stderr, "missing block count\n");
        fclose(file);
        return 1;
    }

    for (uint64_t block_num = 1; block_num <= block_count; ++block_num) {
        uint64_t parsed_num_elems = 0;
        if (read_u64_le(file, &parsed_num_elems) != 0) {
            fprintf(stderr, "missing size for block %llu\n", (unsigned long long)block_num);
            status = 1;
            break;
        }

        if (parsed_num_elems > SIZE_MAX / sizeof(int32_t)) {
            fprintf(stderr, "input block %llu is too large\n", (unsigned long long)block_num);
            status = 1;
            break;
        }

        size_t num_elems = (size_t)parsed_num_elems;
        int32_t* values = malloc(num_elems * sizeof(*values));
        if (values == NULL && num_elems > 0) {
            fprintf(stderr, "failed to allocate block %llu\n", (unsigned long long)block_num);
            status = 1;
            break;
        }

        if (read_exact(file, values, num_elems * sizeof(*values)) != 0) {
            fprintf(stderr, "missing values for block %llu\n", (unsigned long long)block_num);
            free(values);
            status = 1;
            break;
        }

        dot_product_vec_args_t args = {values, values, num_elems};

        printf("num elems: %zu\n", num_elems);

        performance_ret serial = test_dot_product_vec(dot_product_vec, args, ITERATIONS);
        printf("checksum serial: %d\n", serial.checksum);
        printf("elapsed wall serial: %.6f seconds\n", serial.elapsed_seconds);

        performance_ret parallel = test_dot_product_vec(dot_product_vec_parallel, args, ITERATIONS);
        printf("checksum parallel: %d\n", parallel.checksum);
        printf("elapsed wall parallel: %.6f seconds\n\n", parallel.elapsed_seconds);

        free(values);
    }

    fclose(file);
    return status;
}

int read_exact(FILE* file, void* buffer, size_t num_bytes) {
    unsigned char* bytes = buffer;

    while (num_bytes > 0) {
        size_t bytes_read = fread(bytes, 1, num_bytes, file);
        if (bytes_read == 0) {
            return -1;
        }

        bytes += bytes_read;
        num_bytes -= bytes_read;
    }

    return 0;
}

int read_u64_le(FILE* file, uint64_t* value) {
    unsigned char bytes[sizeof(*value)] = {0};
    if (read_exact(file, bytes, sizeof(bytes)) != 0) {
        return -1;
    }

    *value = 0;
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        *value |= (uint64_t)bytes[i] << (i * 8);
    }

    return 0;
}

static inline uint32x4_t dot_product_work_vec(int32x4_t vec_a, int32x4_t vec_b) {
    uint32x4_t x = vreinterpretq_u32_s32(vec_a);
    uint32x4_t y = vreinterpretq_u32_s32(vec_b);
    uint32x4_t product = vmulq_u32(x, y);
    uint32x4_t acc = product;
    uint32x4_t mix = vaddq_u32(x, vdupq_n_u32(0x9e3779b9u));

    for (int round = 0; round < COMPUTE_ROUNDS; ++round) {
        product = vaddq_u32(vmulq_u32(product, vdupq_n_u32(1664525u)), mix);
        mix = vaddq_u32(mix, vaddq_u32(y, vdupq_n_u32(1013904223u + (uint32_t)round)));
        acc = vaddq_u32(acc, product);
    }

    return acc;
}

static inline uint32_t dot_product_work_scalar(int32_t a, int32_t b) {
    uint32_t x = (uint32_t)a;
    uint32_t y = (uint32_t)b;
    uint32_t product = x * y;
    uint32_t acc = product;
    uint32_t mix = x + 0x9e3779b9u;

    for (int round = 0; round < COMPUTE_ROUNDS; ++round) {
        product = product * 1664525u + mix;
        mix += y + 1013904223u + (uint32_t)round;
        acc += product;
    }

    return acc;
}

int32_t dot_product_vec(int32_t* a, int32_t* b, size_t num_elems) {

    size_t i = 0;

    uint32x4_t sum = vdupq_n_u32(0);
    for (; i < (num_elems / 4) * 4; i += 4) {
        int32x4_t vec_a = vld1q_s32(a + i);
        int32x4_t vec_b = vld1q_s32(b + i);

        sum = vaddq_u32(sum, dot_product_work_vec(vec_a, vec_b));
    }

    uint32_t result = vaddvq_u32(sum);
    for (; i < num_elems; ++i) {
        result += dot_product_work_scalar(a[i], b[i]);
    }

    return (int32_t)result;
}

int32_t dot_product_vec_parallel(int32_t* a, int32_t* b, size_t num_elems) {

    omp_set_num_threads(10);

    uint32x4_t sum = vdupq_n_u32(0);
    
    #pragma omp parallel
    {
        uint32x4_t local_sum = vdupq_n_u32(0);

        #pragma omp for schedule(guided)
        for (size_t i = 0; i < (num_elems / 4) * 4; i += 4) {
            int32x4_t vec_a = vld1q_s32(a + i);
            int32x4_t vec_b = vld1q_s32(b + i);

            local_sum = vaddq_u32(local_sum, dot_product_work_vec(vec_a, vec_b));
        }

        #pragma omp critical
        sum = vaddq_u32(sum, local_sum);
    }


    uint32_t result = vaddvq_u32(sum);

    #pragma omp parallel for reduction(+: result)
    for (size_t i = (num_elems / 4) * 4; i < num_elems; ++i) {
        result += dot_product_work_scalar(a[i], b[i]);
    }

    return (int32_t)result;
}

performance_ret test_dot_product_vec(DotProductVec func, dot_product_vec_args_t args, size_t iterations) {
    int64_t checksum = 0;

    double beg = omp_get_wtime();
    for (size_t i = 0; i < iterations; ++i) {
        checksum += func(args.a, args.b, args.num_elems);
    }
    double end = omp_get_wtime();

    double elapsed_seconds = end - beg;

    return (performance_ret){checksum, elapsed_seconds};
}
