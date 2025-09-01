/* E.P 2 - Filtro de Realce Seletivo
> Paralelização em Pthreads

João Pedro Corrêa Silva                 R.A: 11202321629
João Pedro Sousa Bianchim               R.A: 11201920729
Thiago Vinícius Pereira Graciano de Souza   R.A: 11201722589

Professor: Emílio Francesquini
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>

// Structs e Prototipos ---------------------------------------------------

typedef struct {
    unsigned char r, g, b;
} Pixel;

// Argumentos que cada thread do blur precisa saber
typedef struct {
    long thread_id;
    int total_threads;
    Pixel *image_in;
    Pixel *image_out;
    int width;
    int height;
    int M;
} BlurArgs;

// Argumentos para as threads do sharpen
typedef struct {
    long thread_id;
    int total_threads;
    Pixel *image_original;
    Pixel *image_blurred;
    Pixel *image_sharpened;
    long total_pixels;
    int threshold;
    double alpha;
} SharpenArgs;

// Argumentos para as threads do grayscale
typedef struct {
    long thread_id;
    int total_threads;
    Pixel *image_in;
    unsigned char *image_out_gray;
    long total_pixels;
} GrayscaleArgs;

// Funções das threads
void* blur_worker(void* args);
void* sharpen_worker(void* args);
void* grayscale_worker(void* args);

// Funções auxiliares
static inline unsigned char clamp_byte(int v);
int read_next_int(FILE *f, int *out);


// Main --------------------------------------------------------------------

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

    if (M <= 0) { fprintf(stderr, "M precisa ser > 0\n"); return 1; }
    if (thread_count <= 0) { fprintf(stderr, "O número de threads precisa ser > 0\n"); return 1; }

    FILE *fin = fopen(infile, "r");
    if (!fin) { perror("Erro ao abrir arquivo de entrada"); return 1; }

    char magic[3] = {0};
    if (fscanf(fin, "%2s", magic) != 1 || strcmp(magic, "P3") != 0) {
        fprintf(stderr, "Opa, esse filtro só funciona com imagem PPM (P3).\n");
        fclose(fin);
        return 1;
    }

    int width, height, maxval;
    if (!read_next_int(fin, &width) || !read_next_int(fin, &height) || !read_next_int(fin, &maxval)) {
        fprintf(stderr, "Deu ruim lendo o cabeçalho da imagem.\n");
        fclose(fin);
        return 1;
    }

    long total_pixels = (long)width * height;
    Pixel *original_image = malloc(total_pixels * sizeof(Pixel));
    Pixel *blurred_image = malloc(total_pixels * sizeof(Pixel));
    Pixel *sharpened_image = malloc(total_pixels * sizeof(Pixel));
    unsigned char *final_gray_image = malloc(total_pixels * sizeof(unsigned char));

    if (!original_image || !blurred_image || !sharpened_image || !final_gray_image) {
        perror("Faltou memória pra alocar as imagens");
        // Limpa o que já foi alocado
        free(original_image); free(blurred_image); free(sharpened_image); free(final_gray_image);
        return 1;
    }

    for (long i = 0; i < total_pixels; ++i) {
        int r, g, b;
        if (!read_next_int(fin, &r) || !read_next_int(fin, &g) || !read_next_int(fin, &b)) {
            fprintf(stderr, "Erro lendo pixel %ld\n", i); return 1;
        }
        original_image[i].r = clamp_byte(r);
        original_image[i].g = clamp_byte(g);
        original_image[i].b = clamp_byte(b);
    }
    fclose(fin);
    
    // --- Bora paralelizar o trampo ---
    pthread_t* thread_handles = malloc(thread_count * sizeof(pthread_t));

    // Etapa 1: Aplicar o blur com N threads
    BlurArgs* blur_args = malloc(thread_count * sizeof(BlurArgs));
    for (long i = 0; i < thread_count; i++) {
        blur_args[i] = (BlurArgs){i, thread_count, original_image, blurred_image, width, height, M};
        pthread_create(&thread_handles[i], NULL, blur_worker, &blur_args[i]);
    }
    // Espera todo mundo do blur terminar pra continuar
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(blur_args);

    // Etapa 2: Aplicar o sharpen
    SharpenArgs* sharpen_args = malloc(thread_count * sizeof(SharpenArgs));
    for (long i = 0; i < thread_count; i++) {
        sharpen_args[i] = (SharpenArgs){i, thread_count, original_image, blurred_image, sharpened_image, total_pixels, threshold, alpha};
        pthread_create(&thread_handles[i], NULL, sharpen_worker, &sharpen_args[i]);
    }
    // Barreira: espera o sharpen acabar
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(sharpen_args);

    // Etapa 3: Converter pra tons de cinza
    GrayscaleArgs* gray_args = malloc(thread_count * sizeof(GrayscaleArgs));
    for (long i = 0; i < thread_count; i++) {
        gray_args[i] = (GrayscaleArgs){i, thread_count, sharpened_image, final_gray_image, total_pixels};
        pthread_create(&thread_handles[i], NULL, grayscale_worker, &gray_args[i]);
    }
    // Espera a última etapa
    for (long i = 0; i < thread_count; i++) pthread_join(thread_handles[i], NULL);
    free(gray_args);

    free(thread_handles);
    // --- Fim do trabalho pesado ---

    FILE *fo = fopen(outfile, "w");
    if (!fo) { perror("Erro pra criar o arquivo de saída"); return 1; }
    fprintf(fo, "P3\n%d %d\n255\n", width, height);
    for (long i = 0; i < total_pixels; ++i) {
        fprintf(fo, "%d %d %d ", final_gray_image[i], final_gray_image[i], final_gray_image[i]);
        if ((i + 1) % width == 0) fprintf(fo, "\n");
    }
    fclose(fo);

    free(original_image);
    free(blurred_image);
    free(sharpened_image);
    free(final_gray_image);

    printf("Pronto! Imagem '%s' processada com %d threads.\n", outfile, thread_count);
    return 0;
}


// --- Funções das Threads (Workers) ---

void* blur_worker(void* args) {
    BlurArgs* my_args = (BlurArgs*)args;
    long thread_id = my_args->thread_id;
    int total_threads = my_args->total_threads;
    int width = my_args->width;
    int height = my_args->height;

    // Cada thread descobre qual fatia de linhas da imagem ela vai processar.
    int rows_per_thread = height / total_threads;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id == total_threads - 1) 
                    ? height // A última thread pega o resto pra não sobrar pixel.
                    : start_row + rows_per_thread;

    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            long idx = (long)y * width + x;
            // O raio do blur aqui é a parte "chata", ele muda pra cada pixel.
            int rgb_sum = my_args->image_in[idx].r + my_args->image_in[idx].g + my_args->image_in[idx].b;
            int radius = (rgb_sum % my_args->M) + 1;

            long sum_r = 0, sum_g = 0, sum_b = 0;
            int count = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int neighbor_x = x + dx, neighbor_y = y + dy;
                    // Clamp nas bordas pra não ler lixo da memória.
                    if (neighbor_x < 0) neighbor_x = 0;
                    if (neighbor_y < 0) neighbor_y = 0;
                    if (neighbor_x >= width) neighbor_x = width - 1;
                    if (neighbor_y >= height) neighbor_y = height - 1;
                    
                    long neighbor_idx = (long)neighbor_y * width + neighbor_x;
                    sum_r += my_args->image_in[neighbor_idx].r;
                    sum_g += my_args->image_in[neighbor_idx].g;
                    sum_b += my_args->image_in[neighbor_idx].b;
                    count++;
                }
            }
            my_args->image_out[idx].r = clamp_byte((int)round((double)sum_r / count));
            my_args->image_out[idx].g = clamp_byte((int)round((double)sum_g / count));
            my_args->image_out[idx].b = clamp_byte((int)round((double)sum_b / count));
        }
    }
    return NULL;
}

void* sharpen_worker(void* args) {
    SharpenArgs* my_args = (SharpenArgs*)args;
    long thread_id = my_args->thread_id;
    int total_threads = my_args->total_threads;
    long total_pixels = my_args->total_pixels;

    // Aqui a gente divide o vetor de pixels 1D em pedaços (chunks).
    long chunk_size = total_pixels / total_threads;
    long start_index = thread_id * chunk_size;
    long end_index = (thread_id == total_threads - 1) 
                       ? total_pixels // Última thread pega o que sobrou.
                       : start_index + chunk_size;

    for (long i = start_index; i < end_index; ++i) {
        if (my_args->image_original[i].r > my_args->threshold) {
            double new_r = my_args->image_original[i].r + my_args->alpha * (my_args->image_original[i].r - my_args->image_blurred[i].r);
            double new_g = my_args->image_original[i].g + my_args->alpha * (my_args->image_original[i].g - my_args->image_blurred[i].g);
            double new_b = my_args->image_original[i].b + my_args->alpha * (my_args->image_original[i].b - my_args->image_blurred[i].b);
            my_args->image_sharpened[i].r = clamp_byte((int)round(new_r));
            my_args->image_sharpened[i].g = clamp_byte((int)round(new_g));
            my_args->image_sharpened[i].b = clamp_byte((int)round(new_b));
        } else {
            my_args->image_sharpened[i] = my_args->image_blurred[i];
        }
    }
    return NULL;
}

void* grayscale_worker(void* args) {
    GrayscaleArgs* my_args = (GrayscaleArgs*)args;
    long thread_id = my_args->thread_id;
    int total_threads = my_args->total_threads;
    long total_pixels = my_args->total_pixels;

    // Mesma lógica do sharpen: divide o vetorzão de pixels.
    long chunk_size = total_pixels / total_threads;
    long start_index = thread_id * chunk_size;
    long end_index = (thread_id == total_threads - 1) 
                       ? total_pixels 
                       : start_index + chunk_size;

    for (long i = start_index; i < end_index; ++i) {
        int r = my_args->image_in[i].r;
        int g = my_args->image_in[i].g;
        int b = my_args->image_in[i].b;
        double y_val = 0.299 * r + 0.587 * g + 0.114 * b;
        my_args->image_out_gray[i] = clamp_byte((int)round(y_val));
    }
    return NULL;
}


// --- Funções Auxiliares ---
static inline unsigned char clamp_byte(int v) {
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