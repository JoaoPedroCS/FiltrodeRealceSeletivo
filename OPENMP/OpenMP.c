#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h> // Header para OpenMP

// Estrutura para representar um pixel RGB
typedef struct {
    unsigned char r, g, b;
} Pixel;

// Função para garantir que um valor de cor permaneça no intervalo [0, 255]
unsigned char clamp(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

int main(int argc, char *argv[]) {
    // 1. VERIFICAÇÃO E PROCESSAMENTO DOS ARGUMENTOS (SEQUENCIAL)
    if (argc != 6) {
        fprintf(stderr, "Erro: Número incorreto de argumentos.\n");
        fprintf(stderr, "Uso: %s <input.ppm> <output.ppm> <M> <limiar> <sharpen_factor>\n", argv[0]);
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

    // 2. ABERTURA, LEITURA E ALOCAÇÃO (SEQUENCIAL)
    FILE *inputFile = fopen(input_filename, "r");
    if (!inputFile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    char magic[3];
    int width, height, max_val;
    fscanf(inputFile, "%2s %d %d %d", magic, &width, &height, &max_val);
    if (strcmp(magic, "P3") != 0) {
        fprintf(stderr, "Erro: O arquivo de entrada não é um PPM P3 válido.\n");
        fclose(inputFile);
        return 1;
    }

    printf("Lendo imagem '%s' (%d x %d)...\n", input_filename, width, height);
    printf("Parâmetros do filtro: M=%d, limiar=%d, sharpen_factor=%.2f\n", M, limiar, sharpen_factor);
    
    Pixel *original_image = (Pixel *)malloc(width * height * sizeof(Pixel));
    unsigned char *grayscale_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blurred_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    Pixel *final_image = (Pixel *)malloc(width * height * sizeof(Pixel));

    if (!original_image || !grayscale_image || !blurred_image || !final_image) {
        fprintf(stderr, "Erro: Falha ao alocar memória.\n");
        // ... (código de liberação de memória)
        return 1;
    }

    for (int i = 0; i < width * height; i++) {
        fscanf(inputFile, "%hhu %hhu %hhu", &original_image[i].r, &original_image[i].g, &original_image[i].b);
    }
    fclose(inputFile);

    // Medição de tempo do processamento
    double start_time = omp_get_wtime();

    // --------------------------------------------------------------------------
    // ETAPA 1: CONVERSÃO PARA TONS DE CINZA (PARALELO)
    // --------------------------------------------------------------------------
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        grayscale_image[i] = (unsigned char)(0.299 * original_image[i].r + 0.587 * original_image[i].g + 0.114 * original_image[i].b);
    }

    // --------------------------------------------------------------------------
    // ETAPA 2: DESFOQUE DE RAIO VARIÁVEL (PARALELO com schedule dinâmico)
    // --------------------------------------------------------------------------
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int current_pixel_idx = y * width + x;
            Pixel p_orig = original_image[current_pixel_idx];
            int radius = ((p_orig.r + p_orig.g + p_orig.b) % M) + 1;

            long sum = 0;
            int count = 0;
            for (int j = -radius; j <= radius; j++) {
                for (int i = -radius; i <= radius; i++) {
                    int neighbor_x = x + i;
                    int neighbor_y = y + j;
                    if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                        sum += grayscale_image[neighbor_y * width + neighbor_x];
                        count++;
                    }
                }
            }
            blurred_image[current_pixel_idx] = (unsigned char)(sum / count);
        }
    }

    // --------------------------------------------------------------------------
    // ETAPA 3: AJUSTE SELETIVO (PARALELO)
    // --------------------------------------------------------------------------
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        Pixel p_orig = original_image[i];
        unsigned char blurred_val = blurred_image[i];

        if (p_orig.r > limiar) {
            float new_r = p_orig.r + sharpen_factor * (p_orig.r - blurred_val);
            float new_g = p_orig.g + sharpen_factor * (p_orig.g - blurred_val);
            float new_b = p_orig.b + sharpen_factor * (p_orig.b - blurred_val);
            
            final_image[i].r = clamp((int)round(new_r));
            final_image[i].g = clamp((int)round(new_g));
            final_image[i].b = clamp((int)round(new_b));
        } else {
            final_image[i] = original_image[i];
        }
    }

    double end_time = omp_get_wtime();
    printf("Tempo de processamento do filtro: %f segundos\n", end_time - start_time);

    // --------------------------------------------------------------------------
    // 3. ESCRITA DO ARQUIVO DE SAÍDA (SEQUENCIAL)
    // --------------------------------------------------------------------------
    printf("Escrevendo imagem de saída em '%s'...\n", output_filename);
    FILE *outputFile = fopen(output_filename, "w");
    if (!outputFile) {
        perror("Erro ao criar o arquivo de saída");
        // ... (código de liberação de memória)
        return 1;
    }

    fprintf(outputFile, "P3\n%d %d\n%d\n", width, height, max_val);
    for (int i = 0; i < width * height; i++) {
        fprintf(outputFile, "%d %d %d\n", final_image[i].r, final_image[i].g, final_image[i].b);
    }
    fclose(outputFile);

    // 4. LIBERAÇÃO DE MEMÓRIA (SEQUENCIAL)
    free(original_image);
    free(grayscale_image);
    free(blurred_image);
    free(final_image);

    printf("Filtro aplicado com sucesso!\n");

    return 0;
}