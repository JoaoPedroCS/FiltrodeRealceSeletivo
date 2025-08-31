#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char r, g, b;
} Pixel;

void Allocate_memmory(Pixel **d_original_p, unsigned char **d_grayscale_p, unsigned char **d_blurred_p,
                      Pixel **d_final_p, int width, int height) {
    int n = width * height;
    cudaMalloc((void **)d_original_p, n * sizeof(Pixel));
    cudaMalloc((void **)d_grayscale_p, n * sizeof(unsigned char));
    cudaMalloc((void **)d_blurred_p, n * sizeof(unsigned char));
    cudaMalloc((void **)d_final_p, n * sizeof(Pixel));
}

void Free_vectors(Pixel **d_original_p, unsigned char **d_grayscale_p, unsigned char **d_blurred_p,
                  Pixel **d_final_p) {
    cudaFree(*d_original_p);
    cudaFree(*d_grayscale_p);
    cudaFree(*d_blurred_p);
    cudaFree(*d_final_p);
}

__global__ void gray_scale_transformation(const Pixel *orig, unsigned char *gray, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned char r = orig[idx].r;
    unsigned char g = orig[idx].g;
    unsigned char b = orig[idx].b;

    gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

__global__ void blur_transformation(const Pixel *orig, const unsigned char *gray,
                                    unsigned char *blur, int width, int height, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    Pixel p = orig[idx];
    int radius = ((p.r + p.g + p.b) % M) + 1;
    int x = idx % width;
    int y = idx / width;

    int x0 = max(0, x - radius);
    int x1 = min(width - 1, x + radius);
    int y0 = max(0, y - radius);
    int y1 = min(height - 1, y + radius);

    unsigned int sum = 0;
    int count = 0;
    for (int yy = y0; yy <= y1; ++yy) {
        int base = yy * width;
        for (int xx = x0; xx <= x1; ++xx) {
            sum += gray[base + xx];
            ++count;
        }
    }
    blur[idx] = (unsigned char)(sum / (count ? count : 1));
}

__device__ unsigned char clamp(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

__global__ void sharpen_kernel(const Pixel *orig, const unsigned char *blur,
                               Pixel *out, int n, int limiar, float sharpen_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        Pixel p = orig[i];
        unsigned char b = blur[i];

        if (p.r > limiar) {
            float new_r = (float)p.r + sharpen_factor * ((float)p.r - (float)b);
            float new_g = (float)p.g + sharpen_factor * ((float)p.g - (float)b);
            float new_b = (float)p.b + sharpen_factor * ((float)p.b - (float)b);

            out[i].r = clamp((int)new_r);
            out[i].g = clamp((int)new_g);
            out[i].b = clamp((int)new_b);
        } else {
            out[i] = p;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Erro: Número incorreto de argumentos.\n");
        fprintf(stderr, "Uso: %s <input.ppm> <output.ppm> <M> <limiar> <sharpen_factor>\n", argv[0]);
        fprintf(stderr, "Onde:\n");
        fprintf(stderr, "  M:               Inteiro >= 1 para a fórmula do raio.\n");
        fprintf(stderr, "  limiar:          Inteiro [0-255] para o critério de sharpen.\n");
        fprintf(stderr, "  sharpen_factor:  Float para a intensidade do sharpen (ex: 1.2).\n");
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    int M = atoi(argv[3]);
    int limiar = atoi(argv[4]);
    float sharpen_factor = atof(argv[5]);

    if (M < 1) {
        fprintf(stderr, "Erro: M deve ser um inteiro maior ou igual a 1.\n");
        return 1;
    }
    if (limiar < 0 || limiar > 255) {
        fprintf(stderr, "Erro: limiar deve ser um inteiro no intervalo [0, 255].\n");
        return 1;
    }

    // 2. ABERTURA E LEITURA DO ARQUIVO DE ENTRADA
    FILE *inputFile = fopen(input_filename, "r");
    if (!inputFile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    char magic[3];
    int width, height, max_val;
    if (fscanf(inputFile, "%2s", magic) != 1) {
        fprintf(stderr, "Erro lendo o cabeçalho do arquivo.\n");
        fclose(inputFile);
        return 1;
    }
    if (strcmp(magic, "P3") != 0) {
        fprintf(stderr, "Erro: O arquivo de entrada não é um PPM P3 válido.\n");
        fclose(inputFile);
        return 1;
    }

    int c;
    while ((c = fgetc(inputFile)) == ' ' || c == '\t' || c == '\n');
    if (c == '#') {
        while (fgetc(inputFile) != '\n');
    } else {
        ungetc(c, inputFile);
    }

    if (fscanf(inputFile, "%d %d %d", &width, &height, &max_val) != 3) {
        fprintf(stderr, "Erro lendo dimensões PPM.\n");
        fclose(inputFile);
        return 1;
    }

    printf("Lendo imagem '%s' (%d x %d)...\n", input_filename, width, height);
    printf("Parâmetros do filtro: M=%d, limiar=%d, sharpen_factor=%.2f\n", M, limiar, sharpen_factor);

    int n = width * height;

    Pixel *cpu_original_image = (Pixel *)malloc(n * sizeof(Pixel));
    Pixel *cpu_final_image = (Pixel *)malloc(n * sizeof(Pixel));

    for (int i = 0; i < width * height; i++) {
        fscanf(inputFile, "%hhu %hhu %hhu", &cpu_original_image[i].r, &cpu_original_image[i].g, &cpu_original_image[i].b);
    }    
    fclose(inputFile);

    // Device buffers
    Pixel *gpu_original_image, *gpu_final_image;
    unsigned char *gpu_grayscale_image, *gpu_blurred_image;

    Allocate_memmory(&gpu_original_image, &gpu_grayscale_image, &gpu_blurred_image, &gpu_final_image, width, height);

    cudaMemcpy(gpu_original_image, cpu_original_image, n * sizeof(Pixel), cudaMemcpyHostToDevice);

    int th_per_blk = 256;
    int blk_ct = (int)((n + th_per_blk - 1) / th_per_blk);

    gray_scale_transformation<<<blk_ct, th_per_blk>>>(gpu_original_image, gpu_grayscale_image, (int)n);
    cudaDeviceSynchronize();
    printf("Fiz Gray Scale!\n");

    blur_transformation<<<blk_ct, th_per_blk>>>(gpu_original_image, gpu_grayscale_image, gpu_blurred_image, width, height, M);
    cudaDeviceSynchronize();
    printf("Fiz Blur!\n");

    sharpen_kernel<<<blk_ct, th_per_blk>>>(gpu_original_image, gpu_blurred_image, gpu_final_image, (int)n, limiar, sharpen_factor);
    cudaGetLastError();
    cudaDeviceSynchronize();
    printf("Fiz Sharpen!\n");

    // Copy device -> host
    cudaMemcpy(cpu_final_image, gpu_final_image, n * sizeof(Pixel), cudaMemcpyDeviceToHost);

    printf("Escrevendo imagem de saída em '%s'...\n", output_filename);
    FILE *outputFile = fopen(output_filename, "w");
    if (!outputFile) {
        perror("Erro ao criar o arquivo de saída");
        Free_vectors(&gpu_original_image, &gpu_grayscale_image, &gpu_blurred_image, &gpu_final_image);
        free(cpu_original_image);
        free(cpu_final_image);
        return 1;
    }

    fprintf(outputFile, "P3\n%d %d\n%d\n", width, height, max_val);
    for (int i = 0; i < n; i++) {
        fprintf(outputFile, "%u %u %u\n", (unsigned int)cpu_final_image[i].r,
                (unsigned int)cpu_final_image[i].g, (unsigned int)cpu_final_image[i].b);
    }
    fclose(outputFile);

    Free_vectors(&gpu_original_image, &gpu_grayscale_image, &gpu_blurred_image, &gpu_final_image);
    free(cpu_original_image);
    free(cpu_final_image);

    printf("Filtro aplicado com sucesso!\n");
    return 0;
}
