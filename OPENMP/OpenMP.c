/* 
E.P 2 - Filtro de Realce Seletivo
> Paralelização em OpenMP

João Pedro Corrêa Silva	        				R.A: 11202321629
João Pedro Sousa Bianchim		    			R.A: 11201920729
Thiago Vinícius Pereira Graciano de Souza   	R.A: 11201722589

Professor: Emílio Francesquini
*/
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Aqui o inicio é igual do openMP, então sem mais delongas
typedef struct {
    unsigned char r, g, b;
} Pixel;
static inline unsigned char clamp_byte(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

int read_next_int(FILE *f, int *out) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(f)) != EOF && c != '\n');
            continue;
        }
        ungetc(c, f);
        if (fscanf(f, "%d", out) == 1) return 1;
        return 0;
    }
    return 0;
}



int main(int argc, char **argv) {
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "Uso: %s entrada.ppm saida.ppm M limiar fator_sharpen [num_threads]\n", argv[0]);
        return 1;
    }

    const char *infile = argv[1];
    const char *outfile = argv[2];
    int M = atoi(argv[3]);
    int threshold = atoi(argv[4]);
    double alpha = atof(argv[5]);

    if (argc == 7) {
        int num_threads = atoi(argv[6]);
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
    }
    
    if (M <= 0) {
        fprintf(stderr, "Erro: M deve ser um número positivo.\n");
        return 1;
    }

    FILE *fin = fopen(infile, "rb");
    if (!fin) {
        perror("Não foi possível abrir o arquivo de entrada");
        return 1;
    }

    char magic[3];
    if (fscanf(fin, "%2s", magic) != 1 || strcmp(magic, "P3") != 0) {
        fprintf(stderr, "Erro: O formato do arquivo de entrada deve ser PPM (P3).\n");
        fclose(fin);
        return 1;
    }

    int width, height, maxval;
    if (!read_next_int(fin, &width) || !read_next_int(fin, &height) || !read_next_int(fin, &maxval)) {
        fprintf(stderr, "Erro ao ler o cabeçalho da imagem.\n");
        fclose(fin);
        return 1;
    }

    long total_pixels = width * height;
    Pixel *image_original = malloc(total_pixels * sizeof(Pixel));
    Pixel *image_blurred = malloc(total_pixels * sizeof(Pixel));
    Pixel *image_sharpened = malloc(total_pixels * sizeof(Pixel));
    unsigned char *image_final_gray = malloc(total_pixels * sizeof(unsigned char));

    if (!image_original || !image_blurred || !image_sharpened || !image_final_gray) {
        perror("Falha na alocação de memória para as imagens");
        fclose(fin);
        free(image_original);
        free(image_blurred);
        free(image_sharpened);
        free(image_final_gray);
        return 1;
    }

    for (long i = 0; i < total_pixels; ++i) {
        int r, g, b;
        if (!read_next_int(fin, &r) || !read_next_int(fin, &g) || !read_next_int(fin, &b)) {
            fprintf(stderr, "Erro ao ler o pixel %ld.\n", i);
            fclose(fin);
            free(image_original);
            free(image_blurred);
            free(image_sharpened);
            free(image_final_gray);
            return 1;
        }
        image_original[i].r = (unsigned char)r;
        image_original[i].g = (unsigned char)g;
        image_original[i].b = (unsigned char)b;
    }
    fclose(fin);

    // O raio do blur muda a cada pixel, então o trabalho de cada iteração é diferente.
    // O schedule(dynamic) é essencial aqui pra balancear a carga entre as threads
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            long idx = y * width + x;
            int rgb_sum = image_original[idx].r + image_original[idx].g + image_original[idx].b;
            int radius = (rgb_sum % M) + 1;

            long sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int neighbor_x = x + dx;
                    int neighbor_y = y + dy;

                    // Trata as bordas da imagem
                    if (neighbor_x < 0) neighbor_x = 0;
                    if (neighbor_y < 0) neighbor_y = 0;
                    if (neighbor_x >= width) neighbor_x = width - 1;
                    if (neighbor_y >= height) neighbor_y = height - 1;

                    long neighbor_idx = neighbor_y * width + neighbor_x;
                    sum_r += image_original[neighbor_idx].r;
                    sum_g += image_original[neighbor_idx].g;
                    sum_b += image_original[neighbor_idx].b;
                    count++;
                }
            }
            image_blurred[idx].r = clamp_byte((int)round((double)sum_r / count));
            image_blurred[idx].g = clamp_byte((int)round((double)sum_g / count));
            image_blurred[idx].b = clamp_byte((int)round((double)sum_b / count));
        }
    }

    // O sharpen é basico, cada pixel é independente do outro.
    // Um 'parallel for' simples já resolve bem.
    #pragma omp parallel for
    for (long i = 0; i < total_pixels; ++i) {
        if (image_original[i].r > threshold) {
            double new_r = image_original[i].r + alpha * (image_original[i].r - image_blurred[i].r);
            double new_g = image_original[i].g + alpha * (image_original[i].g - image_blurred[i].g);
            double new_b = image_original[i].b + alpha * (image_original[i].b - image_blurred[i].b);
            image_sharpened[i].r = clamp_byte((int)round(new_r));
            image_sharpened[i].g = clamp_byte((int)round(new_g));
            image_sharpened[i].b = clamp_byte((int)round(new_b));
        } else {
            image_sharpened[i] = image_blurred[i];
        }
    }

    // Mesma lógica do sharpen: conversão pra cinza é pixel a pixel.
    // É só dividir o loop entre as threads.
    #pragma omp parallel for
    for (long i = 0; i < total_pixels; ++i) {
        double luminance = 0.299 * image_sharpened[i].r + 0.587 * image_sharpened[i].g + 0.114 * image_sharpened[i].b;
        image_final_gray[i] = clamp_byte((int)round(luminance));
    }

    FILE *fout = fopen(outfile, "w");
    if (!fout) {
        perror("Não foi possível criar o arquivo de saída");
        free(image_original);
        free(image_blurred);
        free(image_sharpened);
        free(image_final_gray);
        return 1;
    }

    fprintf(fout, "P3\n%d %d\n255\n", width, height);
    for (long i = 0; i < total_pixels; ++i) {
        fprintf(fout, "%d %d %d ", image_final_gray[i], image_final_gray[i], image_final_gray[i]);
        if ((i + 1) % width == 0) {
            fprintf(fout, "\n");
        }
    }
    fclose(fout);

    free(image_original);
    free(image_blurred);
    free(image_sharpened);
    free(image_final_gray);

    return 0;
}