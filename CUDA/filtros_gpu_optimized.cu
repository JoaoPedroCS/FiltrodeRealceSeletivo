/* 
E.P 2 - Filtro de Realce Seletivo
 > Paralelização em CUDA

João Pedro Corrêa Silva	        				R.A: 11202321629
João Pedro Sousa Bianchim		    			R.A: 11201920729
Thiago Vinícius Pereira Graciano de Souza   	R.A: 11201722589

Professor: Emílio Francesquini
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char r, g, b;
} Pixel;

// Função que Aloca Memória na GPU
void Allocate_memmory(Pixel **gpu_original_p, Pixel **gpu_blurred_p, Pixel **gpu_final_p, int width, int height) {
    int n = width * height;
    cudaMalloc((void **)gpu_original_p, n * sizeof(Pixel));
    // gpu_blurred_p armazena a imagem borrada por canal (R,G,B) conforme especificação
    cudaMalloc((void **)gpu_blurred_p, n * sizeof(Pixel));
    cudaMalloc((void **)gpu_final_p, n * sizeof(Pixel));
}

// Função que Libera a memória alocada na GPU
void Free_vectors(Pixel **gpu_original_p, Pixel **gpu_blurred_p, Pixel **gpu_final_p) {
    cudaFree(*gpu_original_p);
    cudaFree(*gpu_blurred_p);
    cudaFree(*gpu_final_p);
}

// Função que executa o cálculo para transformação em gray scale (cada thread processa um pixel)
__global__ void gray_scale_transformation(const Pixel *in, Pixel *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned char r = in[idx].r;
    unsigned char g = in[idx].g;
    unsigned char b = in[idx].b;
    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    out[idx].r = gray;
    out[idx].g = gray;
    out[idx].b = gray;
}

// Função que executa o cálculo para transformação de blur (cada thread processa um pixel)
__global__ void blur_transformation(const Pixel *orig, Pixel *blur, int width, int height, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    Pixel p = orig[idx];
    int radius = ((int)p.r + (int)p.g + (int)p.b) % M + 1;
    int x = idx % width;
    int y = idx / width;

    int x0 = max(0, x - radius);
    int x1 = min(width - 1, x + radius);
    int y0 = max(0, y - radius);
    int y1 = min(height - 1, y + radius);

    unsigned int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;
    for (int yy = y0; yy <= y1; ++yy) {
        int base = yy * width;
        for (int xx = x0; xx <= x1; ++xx) {
            Pixel q = orig[base + xx];
            sumR += q.r;
            sumG += q.g;
            sumB += q.b;
            ++count;
        }
    }
    if (count == 0) count = 1;
    Pixel out;
    out.r = (unsigned char)(sumR / count);
    out.g = (unsigned char)(sumG / count);
    out.b = (unsigned char)(sumB / count);
    blur[idx] = out;
}

// Função para fazer o clamp e impedir valores diferentes dos permitidos para as cores
__device__ unsigned char clamp(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

// Função que executa o cálculo para transformação de sharpen (cada thread processa um pixel)
__global__ void sharpen_kernel(const Pixel *orig, const Pixel *blur,
                               Pixel *out, int n, int limiar, float sharpen_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        Pixel p = orig[i];
        Pixel b = blur[i];

        if (p.r > limiar) {
            float new_r = (float)p.r + sharpen_factor * ((float)p.r - (float)b.r);
            float new_g = (float)p.g + sharpen_factor * ((float)p.g - (float)b.g);
            float new_b = (float)p.b + sharpen_factor * ((float)p.b - (float)b.b);

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
        printf("Erro: Número incorreto de argumentos.\n");
        return 1;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    int M = atoi(argv[3]);
    int limiar = atoi(argv[4]);
    float sharpen_factor = atof(argv[5]);

    // Abertura do arquivo
    if (M < 1) {
        fprintf(stderr, "Erro: M deve ser um inteiro maior ou igual a 1.\n");
        return 1;
    }
    if (limiar < 0 || limiar > 255) {
        fprintf(stderr, "Erro: limiar deve ser um inteiro no intervalo [0, 255].\n");
        return 1;
    }
    FILE *inputFile = fopen(input_filename, "r");
    if (!inputFile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    char magic[3];
    int width, height, max_val;
    if (fscanf(inputFile, "%2s\n%d %d\n%d\n", magic, &width, &height, &max_val) != 4) {
        fprintf(stderr, "Erro lendo dimensões PPM.\n");
        fclose(inputFile);
        return 1;
    }

    int n = width * height;

    // Criação das matrizes para armazenar a imagem na CPU
    Pixel *cpu_original_image = (Pixel *)malloc(n * sizeof(Pixel));
    Pixel *cpu_final_image = (Pixel *)malloc(n * sizeof(Pixel));

    // Leitura do arquivo
    for (int i = 0; i < width * height; i++) {
        fscanf(inputFile, "%hhu %hhu %hhu", &cpu_original_image[i].r, &cpu_original_image[i].g, &cpu_original_image[i].b);
    }    
    fclose(inputFile);

    // Cálculos na GPU
    Pixel *gpu_original_image, *gpu_blurred_image, *gpu_final_image;

    Allocate_memmory(&gpu_original_image, &gpu_blurred_image, &gpu_final_image, width, height);

    cudaMemcpy(gpu_original_image, cpu_original_image, n * sizeof(Pixel), cudaMemcpyHostToDevice);

    int th_per_blk = 256;
    int blk_ct = (int)((n + th_per_blk - 1) / th_per_blk);

    // Esteira de Aplicação dos Filtros
    blur_transformation<<<blk_ct, th_per_blk>>>(gpu_original_image, gpu_blurred_image, width, height, M);
    cudaDeviceSynchronize();

    sharpen_kernel<<<blk_ct, th_per_blk>>>(gpu_original_image, gpu_blurred_image, gpu_final_image, (int)n, limiar, sharpen_factor);
    cudaDeviceSynchronize();

    // Aplica escala de cinza como última transformação
    gray_scale_transformation<<<blk_ct, th_per_blk>>>(gpu_final_image, gpu_final_image, (int)n);
    cudaDeviceSynchronize();

    cudaMemcpy(cpu_final_image, gpu_final_image, n * sizeof(Pixel), cudaMemcpyDeviceToHost);

    //Escrita do Arquivo Final
    FILE *outputFile = fopen(output_filename, "w");
    if (!outputFile) {
        perror("Erro ao criar o arquivo de saída");
        //Liberação de Memória
        Free_vectors(&gpu_original_image, &gpu_blurred_image, &gpu_final_image);
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

    //Liberação de Memória
    Free_vectors(&gpu_original_image, &gpu_blurred_image, &gpu_final_image);
    free(cpu_original_image);
    free(cpu_final_image);

    return 0;
}
