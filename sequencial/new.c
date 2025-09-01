// filtro_sequencial_correto.c
// Implementação sequencial do filtro de realce seletivo conforme o enunciado ep2.pdf
// Uso: ./filtro_seq_correto entrada.ppm saida.ppm M limiar sharpen_factor
// Ex:  ./filtro_seq_correto img_in.ppm img_out.ppm 7 180 1.2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

// Estrutura para um pixel colorido
typedef struct {
    unsigned char r, g, b;
} Pixel;

static inline unsigned char clamp_char(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

// Lê o próximo inteiro do arquivo, pulando comentários '#' e espaços em branco
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
        return 0; // Falha na leitura
    }
    return 0; // Fim do arquivo
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Uso: %s entrada.ppm saida.ppm M limiar sharpen_factor\n", argv[0]);
        return 1;
    }

    const char *infile = argv[1];
    const char *outfile = argv[2];
    int M = atoi(argv[3]);
    int threshold = atoi(argv[4]);
    double alpha = atof(argv[5]);

    if (M <= 0) { fprintf(stderr, "M deve ser > 0\n"); return 1; }

    FILE *fin = fopen(infile, "r");
    if (!fin) { perror("fopen entrada"); return 1; }

    char magic[3] = {0};
    if (fscanf(fin, "%2s", magic) != 1 || strcmp(magic, "P3") != 0) {
        fprintf(stderr, "Formato de entrada deve ser P3 (ASCII PPM)\n");
        fclose(fin);
        return 1;
    }

    int width, height, maxval;
    if (!read_next_int(fin, &width) || !read_next_int(fin, &height) || !read_next_int(fin, &maxval)) {
        fprintf(stderr, "Erro lendo cabeçalho PPM (width/height/maxval)\n");
        fclose(fin);
        return 1;
    }
    if (width <= 0 || height <= 0 || maxval <= 0) {
        fprintf(stderr, "Dimensões ou maxval inválidos\n");
        fclose(fin);
        return 1;
    }

    long npix = width * height;
    Pixel *original_image = malloc(npix * sizeof(Pixel));
    Pixel *blurred_image = malloc(npix * sizeof(Pixel));
    Pixel *sharpened_image = malloc(npix * sizeof(Pixel));
    unsigned char *final_gray = malloc(npix * sizeof(unsigned char));

    if (!original_image || !blurred_image || !sharpened_image || !final_gray) {
        perror("malloc");
        fclose(fin);
        // Libera o que foi alocado com sucesso
        free(original_image); free(blurred_image); free(sharpened_image); free(final_gray);
        return 1;
    }

    // Ler pixels e normalizar se maxval != 255
    for (long i = 0; i < npix; ++i) {
        int r, g, b;
        if (!read_next_int(fin, &r) || !read_next_int(fin, &g) || !read_next_int(fin, &b)) {
            fprintf(stderr, "Erro lendo pixel %ld\n", i);
            // ... (código de liberação de memória)
            return 1;
        }
        if (maxval != 255) {
            r = (int)round(r * 255.0 / maxval);
            g = (int)round(g * 255.0 / maxval);
            b = (int)round(b * 255.0 / maxval);
        }
        original_image[i].r = clamp_char(r);
        original_image[i].g = clamp_char(g);
        original_image[i].b = clamp_char(b);
    }
    fclose(fin);

    // ETAPA 2: Blur de raio variável em cada canal de cor (R,G,B)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            long idx = y * width + x;
            int S = original_image[idx].r + original_image[idx].g + original_image[idx].b;
            int radius = (S % M) + 1;

            long sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    // Tratamento de borda: clamp (replicar a borda)
                    if (nx < 0) nx = 0;
                    if (ny < 0) ny = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny >= height) ny = height - 1;

                    long n_idx = ny * width + nx;
                    sum_r += original_image[n_idx].r;
                    sum_g += original_image[n_idx].g;
                    sum_b += original_image[n_idx].b;
                    count++;
                }
            }
            blurred_image[idx].r = clamp_char((int)round((double)sum_r / count));
            blurred_image[idx].g = clamp_char((int)round((double)sum_g / count));
            blurred_image[idx].b = clamp_char((int)round((double)sum_b / count));
        }
    }

    // ETAPA 3: Sharpen seletivo em cada canal de cor (R,G,B)
    for (long i = 0; i < npix; ++i) {
        if (original_image[i].r > threshold) {
            double new_r = original_image[i].r + alpha * (original_image[i].r - blurred_image[i].r);
            double new_g = original_image[i].g + alpha * (original_image[i].g - blurred_image[i].g);
            double new_b = original_image[i].b + alpha * (original_image[i].b - blurred_image[i].b);
            sharpened_image[i].r = clamp_char((int)round(new_r));
            sharpened_image[i].g = clamp_char((int)round(new_g));
            sharpened_image[i].b = clamp_char((int)round(new_b));
        } else {
            sharpened_image[i] = blurred_image[i];
        }
    }

    // ETAPA 4: Conversão final para tons de cinza
    for (long i = 0; i < npix; ++i) {
        int r = sharpened_image[i].r;
        int g = sharpened_image[i].g;
        int b = sharpened_image[i].b;
        double y_val = 0.299 * r + 0.587 * g + 0.114 * b;
        final_gray[i] = clamp_char((int)round(y_val));
    }

    // Escreve o arquivo de saída P3 com R=G=B = valor de cinza
    FILE *fo = fopen(outfile, "w");
    if (!fo) {
        perror("fopen saida");
        // ... (código de liberação de memória)
        return 1;
    }
    fprintf(fo, "P3\n%d %d\n255\n", width, height);
    for (long i = 0; i < npix; ++i) {
        fprintf(fo, "%d %d %d ", final_gray[i], final_gray[i], final_gray[i]);
        if ((i + 1) % width == 0) {
            fprintf(fo, "\n");
        }
    }
    fclose(fo);

    // Liberação de memória
    free(original_image);
    free(blurred_image);
    free(sharpened_image);
    free(final_gray);

    return 0;
}