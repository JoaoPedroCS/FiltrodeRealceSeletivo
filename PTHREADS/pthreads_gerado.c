// filtro_pthreads.c
// Implementação paralela com Pthreads do filtro de realce seletivo.
// Uso: ./filtro_pthreads entrada.ppm saida.ppm M limiar sharpen_factor num_threads
// Ex:  ./filtro_pthreads in.ppm out.ppm 7 180 1.2 4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>

// Estrutura para um pixel colorido
typedef struct {
    unsigned char r, g, b;
} Pixel;

// Estruturas de argumentos para cada função de thread
typedef struct {
    long rank;
    int thread_count;
    Pixel *original_image;
    Pixel *blurred_image;
    int width;
    int height;
    int M;
} blur_args_t;

typedef struct {
    long rank;
    int thread_count;
    Pixel *original_image;
    Pixel *blurred_image;
    Pixel *sharpened_image;
    int width;
    int height;
    int limiar;
    double alpha;
} sharpen_args_t;

typedef struct {
    long rank;
    int thread_count;
    Pixel *sharpened_image;
    unsigned char *final_gray_output;
    long npix;
} grayscale_args_t;

// Funções auxiliares e de thread
static inline unsigned char clamp_char(int v);
int read_next_int(FILE *f, int *out);
void* apply_blur_thread(void* args);
void* apply_sharpen_thread(void* args);
void* convert_to_grayscale_thread(void* args);


int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "Uso: %s entrada.ppm saida.ppm M limiar sharpen_factor num_threads\n", argv[0]);
        return 1;
    }

    const char *infile = argv[1];
    const char *outfile = argv[2];
    int M = atoi(argv[3]);
    int threshold = atoi(argv[4]);
    double alpha = atof(argv[5]);
    int thread_count = atoi(argv[6]);

    if (M <= 0) { fprintf(stderr, "M deve ser > 0\n"); return 1; }
    if (thread_count <= 0) { fprintf(stderr, "Número de threads deve ser > 0\n"); return 1; }

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
        fprintf(stderr, "Erro lendo cabeçalho PPM\n");
        fclose(fin);
        return 1;
    }

    long npix = width * height;
    Pixel *original_image = malloc(npix * sizeof(Pixel));
    Pixel *blurred_image = malloc(npix * sizeof(Pixel));
    Pixel *sharpened_image = malloc(npix * sizeof(Pixel));
    unsigned char *final_gray = malloc(npix * sizeof(unsigned char));

    if (!original_image || !blurred_image || !sharpened_image || !final_gray) {
        perror("malloc"); return 1;
    }

    for (long i = 0; i < npix; ++i) {
        int r, g, b;
        if (!read_next_int(fin, &r) || !read_next_int(fin, &g) || !read_next_int(fin, &b)) {
            fprintf(stderr, "Erro lendo pixel %ld\n", i); return 1;
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
    
    // --- Início do Processamento Paralelo ---
    pthread_t* thread_handles = malloc(thread_count * sizeof(pthread_t));

    // ETAPA 1: Blur
    blur_args_t* blur_args = malloc(thread_count * sizeof(blur_args_t));
    for (long i = 0; i < thread_count; i++) {
        blur_args[i] = (blur_args_t){i, thread_count, original_image, blurred_image, width, height, M};
        pthread_create(&thread_handles[i], NULL, apply_blur_thread, (void *)&blur_args[i]);
    }
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(blur_args);

    // ETAPA 2: Sharpen
    sharpen_args_t* sharpen_args = malloc(thread_count * sizeof(sharpen_args_t));
    for (long i = 0; i < thread_count; i++) {
        sharpen_args[i] = (sharpen_args_t){i, thread_count, original_image, blurred_image, sharpened_image, width, height, threshold, alpha};
        pthread_create(&thread_handles[i], NULL, apply_sharpen_thread, (void *)&sharpen_args[i]);
    }
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(sharpen_args);

    // ETAPA 3: Grayscale
    grayscale_args_t* gray_args = malloc(thread_count * sizeof(grayscale_args_t));
    for (long i = 0; i < thread_count; i++) {
        gray_args[i] = (grayscale_args_t){i, thread_count, sharpened_image, final_gray, npix};
        pthread_create(&thread_handles[i], NULL, convert_to_grayscale_thread, (void *)&gray_args[i]);
    }
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(gray_args);

    free(thread_handles);
    // --- Fim do Processamento Paralelo ---

    // Escrita do arquivo de saída
    FILE *fo = fopen(outfile, "w");
    if (!fo) { perror("fopen saida"); return 1; }
    fprintf(fo, "P3\n%d %d\n255\n", width, height);
    for (long i = 0; i < npix; ++i) {
        fprintf(fo, "%d %d %d ", final_gray[i], final_gray[i], final_gray[i]);
        if ((i + 1) % width == 0) fprintf(fo, "\n");
    }
    fclose(fo);

    free(original_image);
    free(blurred_image);
    free(sharpened_image);
    free(final_gray);

    printf("Filtro Pthreads aplicado com sucesso em '%s'.\n", outfile);
    return 0;
}


// --- Implementação das Funções de Thread ---

void* apply_blur_thread(void* args) {
    blur_args_t* p_args = (blur_args_t*)args;
    long my_rank = p_args->rank;
    int thread_count = p_args->thread_count;
    int width = p_args->width;
    int height = p_args->height;

    // Divisão de trabalho por linhas
    int rows_per_thread = height / thread_count;
    int my_first_row = my_rank * rows_per_thread;
    int my_last_row = (my_rank == thread_count - 1) ? height : (my_rank + 1) * rows_per_thread;

    for (int y = my_first_row; y < my_last_row; ++y) {
        for (int x = 0; x < width; ++x) {
            long idx = (long)y * width + x;
            int S = p_args->original_image[idx].r + p_args->original_image[idx].g + p_args->original_image[idx].b;
            int radius = (S % p_args->M) + 1;

            long sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0) nx = 0;
                    if (ny < 0) ny = 0;
                    if (nx >= width) nx = width - 1;
                    if (ny >= height) ny = height - 1;
                    
                    long n_idx = (long)ny * width + nx;
                    sum_r += p_args->original_image[n_idx].r;
                    sum_g += p_args->original_image[n_idx].g;
                    sum_b += p_args->original_image[n_idx].b;
                    count++;
                }
            }
            p_args->blurred_image[idx].r = clamp_char((int)round((double)sum_r / count));
            p_args->blurred_image[idx].g = clamp_char((int)round((double)sum_g / count));
            p_args->blurred_image[idx].b = clamp_char((int)round((double)sum_b / count));
        }
    }
    return NULL;
}

void* apply_sharpen_thread(void* args) {
    sharpen_args_t* p_args = (sharpen_args_t*)args;
    long my_rank = p_args->rank;
    int thread_count = p_args->thread_count;

    // Divisão de trabalho pelo array 1D de pixels
    long long npix = (long long)p_args->width * p_args->height;
    long long chunk_size = npix / thread_count;
    long long my_first_i = my_rank * chunk_size;
    long long my_last_i = (my_rank == thread_count - 1) ? npix : (my_rank + 1) * chunk_size;

    for (long long i = my_first_i; i < my_last_i; ++i) {
        if (p_args->original_image[i].r > p_args->limiar) {
            double new_r = p_args->original_image[i].r + p_args->alpha * (p_args->original_image[i].r - p_args->blurred_image[i].r);
            double new_g = p_args->original_image[i].g + p_args->alpha * (p_args->original_image[i].g - p_args->blurred_image[i].g);
            double new_b = p_args->original_image[i].b + p_args->alpha * (p_args->original_image[i].b - p_args->blurred_image[i].b);
            p_args->sharpened_image[i].r = clamp_char((int)round(new_r));
            p_args->sharpened_image[i].g = clamp_char((int)round(new_g));
            p_args->sharpened_image[i].b = clamp_char((int)round(new_b));
        } else {
            p_args->sharpened_image[i] = p_args->blurred_image[i];
        }
    }
    return NULL;
}

void* convert_to_grayscale_thread(void* args) {
    grayscale_args_t* p_args = (grayscale_args_t*)args;
    long my_rank = p_args->rank;
    int thread_count = p_args->thread_count;

    // Divisão de trabalho pelo array 1D de pixels
    long long chunk_size = p_args->npix / thread_count;
    long long my_first_i = my_rank * chunk_size;
    long long my_last_i = (my_rank == thread_count - 1) ? p_args->npix : (my_rank + 1) * chunk_size;

    for (long long i = my_first_i; i < my_last_i; ++i) {
        int r = p_args->sharpened_image[i].r;
        int g = p_args->sharpened_image[i].g;
        int b = p_args->sharpened_image[i].b;
        double y_val = 0.299 * r + 0.587 * g + 0.114 * b;
        p_args->final_gray_output[i] = clamp_char((int)round(y_val));
    }
    return NULL;
}

// --- Funções Auxiliares ---

static inline unsigned char clamp_char(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
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