#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

//contadores para as barreiras de cada etapa
int counter1 = 0;
int counter2 = 0;
int counter3 = 0;
pthread_mutex_t barrier_mutex;

// Estrutura para representar um pixel RGB
typedef struct {
    unsigned char r, g, b;
} Pixel;

struct thread_cinza{
    int rank;
    int thread_count;
    //Pixel *grayscale_image;
    unsigned char *final_gray_output;
    //Pixel *original_image;
    Pixel *sharpened_image;
    int width;
    int height;
};

struct thread_blur{
    int rank;
    int thread_count;
    Pixel *blurred_image;
    //Pixel *grayscale_image;
    Pixel *original_image;
    int width;
    int height;
    int M;
};

struct thread_sharpen{
    int rank;
    int thread_count;
    Pixel *blurred_image;
    Pixel *original_image;
    Pixel *sharpened_image;
    int width;
    int height;
    int limiar;
    float sharpen_factor;
};

// Função para garantir que um valor de cor permaneça no intervalo [0, 255]
unsigned char clamp(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

void *cinza(void* input) {
    //printf("entrou na função\n");

    long my_rank = (long)((struct thread_cinza*)input)->rank;
    int width = ((struct thread_cinza*)input)->width;
    int height = ((struct thread_cinza*)input)->height;
    int i, j;
    int rows_per_thread = (width*height)/((struct thread_cinza*)input)->thread_count;
    int my_first_row = my_rank*rows_per_thread;
    int my_last_row = (my_rank+1)*rows_per_thread - 1;

    for (i = my_first_row; i <= my_last_row; i++) {
        unsigned char media = (0.299 * ((struct thread_cinza*)input)->sharpened_image[i].r + 0.587 * ((struct thread_cinza*)input)->sharpened_image[i].g + 0.114 * ((struct thread_cinza*)input)->sharpened_image[i].b);
        /*((struct thread_cinza*)input)->grayscale_image[i].r = media;
        ((struct thread_cinza*)input)->grayscale_image[i].g = media;
        ((struct thread_cinza*)input)->grayscale_image[i].b = media;*/
        ((struct thread_cinza*)input)->final_gray_output[i] = media;
    }

    return NULL;
}

void *blur(void* input){
    long my_rank = (long)((struct thread_blur*)input)->rank;
    int width = ((struct thread_blur*)input)->width;
    int height = ((struct thread_blur*)input)->height;
    int thread_count = ((struct thread_cinza*)input)->thread_count;

    //divisão de linhas
    int rows_per_thread = height/thread_count;
    int my_first_row = my_rank * rows_per_thread;
    int my_last_row = (my_rank+1)*rows_per_thread - 1;

    for (int y = my_first_row; y < my_last_row; y++) {
        for (int x = 0; x < width; x++) {
            int current_pixel_idx = y * width + x;
            Pixel p_orig = ((struct thread_blur*)input)->original_image[current_pixel_idx];

            // A fórmula do raio agora usa o parâmetro M
            int radius = ((p_orig.r + p_orig.g + p_orig.b) % ((struct thread_blur*)input)->M) + 1;

            //long sum = 0;
            Pixel sum;
            sum.r=0;
            sum.g=0;
            sum.b=0;
            int count = 0;
            unsigned char temp = 'a';
            for (int j = -radius; j <= radius; j++) {
                for (int i = -radius; i <= radius; i++) {
                    int neighbor_x = x + i;
                    int neighbor_y = y + j;
                    if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                        //sum += ((struct thread_cinza*)input)->grayscale_image[neighbor_y * width + neighbor_x];
                        sum.r+= ((struct thread_blur*)input)->original_image[neighbor_y * width + neighbor_x].r;
                        sum.g+= ((struct thread_blur*)input)->original_image[neighbor_y * width + neighbor_x].g;
                        sum.b+= ((struct thread_blur*)input)->original_image[neighbor_y * width + neighbor_x].b;
                        count++;
                    }
                }
            }
            ((struct thread_blur*)input)->blurred_image[current_pixel_idx].r = (sum.r / count);//(unsigned char)(sum.r / count);
            ((struct thread_blur*)input)->blurred_image[current_pixel_idx].g = (sum.g / count);
            ((struct thread_blur*)input)->blurred_image[current_pixel_idx].b = (sum.b / count);
        }
    }

    return NULL;
}

void* sharpen(void* input){
    long my_rank = (long)((struct thread_sharpen*)input)->rank;
    int width = ((struct thread_sharpen*)input)->width;
    int height = ((struct thread_sharpen*)input)->height;
    int limiar = ((struct thread_sharpen*)input)->limiar;
    float sharpen_factor = ((struct thread_sharpen*)input)->sharpen_factor;
    int thread_count = ((struct thread_cinza*)input)->thread_count;

    //divisão de linhas
    int rows_per_thread = height/thread_count;
    int my_first_row = my_rank * rows_per_thread;
    int my_last_row = (my_rank+1)*rows_per_thread - 1;

    for (int i = my_first_row; i < width * my_last_row; i++) {
        Pixel p_orig = ((struct thread_sharpen*)input)->original_image[i];
        Pixel blurred_val = ((struct thread_sharpen*)input)->blurred_image[i];

        // O critério e o fator de sharpen agora usam os parâmetros limiar e sharpen_factor
        if (p_orig.r > limiar || p_orig.g > limiar || p_orig.b > limiar) {
            if(p_orig.r > limiar){
                float new_r = p_orig.r + sharpen_factor * (p_orig.r - blurred_val.r);
                ((struct thread_sharpen*)input)->sharpened_image[i].r = clamp((int)round(new_r));
            }
            if(p_orig.g > limiar){
                float new_g = p_orig.g + sharpen_factor * (p_orig.g - blurred_val.g);
                ((struct thread_sharpen*)input)->sharpened_image[i].g = clamp((int)round(new_g));
            }
            if(p_orig.b > limiar){
                float new_b = p_orig.b + sharpen_factor * (p_orig.b - blurred_val.b);
                ((struct thread_sharpen*)input)->sharpened_image[i].b = clamp((int)round(new_b));
            }

        } else {
            ((struct thread_sharpen*)input)->sharpened_image[i] = ((struct thread_sharpen*)input)->original_image[i];
        }
    }

    return NULL;
}


int main(int argc, char *argv[]) {
    // 1. VERIFICAÇÃO E PROCESSAMENTO DOS ARGUMENTOS DE LINHA DE COMANDO
    if (argc != 7) {
        fprintf(stderr, "Erro: Número incorreto de argumentos.\n");
        fprintf(stderr, "Uso: %s <input.ppm> <output.ppm> <M> <limiar> <sharpen_factor>\n", argv[0]);
        fprintf(stderr, "Onde:\n");
        fprintf(stderr, "  M:               Inteiro >= 1 para a fórmula do raio.\n");
        fprintf(stderr, "  limiar:          Inteiro [0-255] para o critério de sharpen.\n");
        fprintf(stderr, "  sharpen_factor:  Float para a intensidade do sharpen (ex: 1.2).\n");
        fprintf(stderr, "  thread_count:  Inteiro >= 1 para o número de threads usados.\n");
        return 1;
    }

    // Leitura dos nomes dos arquivos e parâmetros
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    int M = atoi(argv[3]);
    int limiar = atoi(argv[4]);
    float sharpen_factor = atof(argv[5]);
    int thread_count = atoi(argv[6]);//simulando número de cores

    // Validação dos parâmetros numéricos
    if (M < 1) {
        fprintf(stderr, "Erro: M deve ser um inteiro maior ou igual a 1.\n");
        return 1;
    }
    if (limiar < 0 || limiar > 255) {
        fprintf(stderr, "Erro: limiar deve ser um inteiro no intervalo [0, 255].\n");
        return 1;
    }
    if(thread_count < 1){
        fprintf(stderr, "Erro: É necessário pelo menos 1 thread.\n");
        return 1;
    }

    // 2. ABERTURA E LEITURA DO ARQUIVO DE ENTRADA
    FILE *inputFile = fopen(input_filename, "r");
    if (!inputFile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    // Leitura do cabeçalho PPM
    char magic[3];
    int width, height, max_val;
    fscanf(inputFile, "%2s", magic);
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

    fscanf(inputFile, "%d %d %d", &width, &height, &max_val);

    printf("Lendo imagem '%s' (%d x %d)...\n", input_filename, width, height);
    printf("Parâmetros do filtro: M=%d, limiar=%d, sharpen_factor=%.2f\n", M, limiar, sharpen_factor);
    printf("Número de threads: %d\n", thread_count);

    // Alocação de memória para as imagens
    Pixel *original_image = (Pixel *)malloc(width * height * sizeof(Pixel));
    //Pixel *grayscale_image = (Pixel *)malloc(width * height * sizeof(Pixel));
    unsigned char *final_gray = malloc(width * height * sizeof(unsigned char));
    Pixel *blurred_image = (Pixel *)malloc(width * height * sizeof(Pixel));
    Pixel *sharpened_image = (Pixel *)malloc(width * height * sizeof(Pixel));

    if (!original_image || !final_gray || !blurred_image || !sharpened_image) {
        fprintf(stderr, "Erro: Falha ao alocar memória.\n");
        free(original_image); free(final_gray); free(blurred_image); free(sharpened_image);
        fclose(inputFile);
        return 1;
    }

    for (int i = 0; i < width * height; i++) {
        fscanf(inputFile, "%hhu %hhu %hhu", &original_image[i].r, &original_image[i].g, &original_image[i].b);
    }
    fclose(inputFile);

    //variáveis para pthreads
    long thread;
    pthread_t* thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
    

    // --------------------------------------------------------------------------
    // ETAPA 1: DESFOQUE DE RAIO VARIÁVEL (VARIABLE-RADIUS BLUR)
    // --------------------------------------------------------------------------
    printf("Etapa 1: Aplicando desfoque de raio variável...\n");
    
    //struct thread_blur args_blur;// = (struct thread_blur *)malloc(sizeof(struct thread_blur));
    
    struct thread_blur *args_blur = malloc(thread_count * sizeof(struct thread_blur));
    
    for (thread = 0; thread < thread_count; thread++){
        args_blur[thread].blurred_image = blurred_image;
        //args_blur[thread].final_gray_output = final_gray;
        args_blur[thread].original_image = original_image;
        args_blur[thread].width = width;
        args_blur[thread].height = height; 
        args_blur[thread].M = M;
        args_blur[thread].thread_count = thread_count;
        args_blur[thread].rank = thread;
        //printf("pronto pra criar threads\n");
        pthread_create(&thread_handles[thread], NULL,
        blur, (void *)&args_blur);
    }

    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    free(args_blur);
    

    // --------------------------------------------------------------------------
    // ETAPA 2: AJUSTE SELETIVO (SHARPEN)
    // --------------------------------------------------------------------------
    printf("Etapa 2: Aplicando ajuste seletivo (sharpen)...\n");

    struct thread_sharpen *args_sharpen = malloc(thread_count * sizeof(struct thread_sharpen));// = (struct thread_sharpen *)malloc(sizeof(struct thread_sharpen));
    
    
    
    for (thread = 0; thread < thread_count; thread++){
        args_sharpen[thread].rank = thread;
        args_sharpen[thread].blurred_image = blurred_image;
        args_sharpen[thread].original_image = original_image;
        args_sharpen[thread].sharpened_image = sharpened_image;
        args_sharpen[thread].width = width;
        args_sharpen[thread].height = height; 
        args_sharpen[thread].limiar = limiar;
        args_sharpen[thread].sharpen_factor = sharpen_factor;
        args_sharpen[thread].thread_count = thread_count;
        //printf("pronto pra criar threads\n");
        pthread_create(&thread_handles[thread], NULL,
        sharpen, (void *)&args_sharpen);
    }

    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    free(args_sharpen);
    // --------------------------------------------------------------------------
    // ETAPA 3: CONVERSÃO PARA TONS DE CINZA (GRAYSCALE)
    // --------------------------------------------------------------------------
    printf("Etapa 3: Convertendo para tons de cinza...\n");

    /*
    for (int i = 0; i < width * height; i++) {
        grayscale_image[i] = (unsigned char)(0.299 * original_image[i].r + 0.587 * original_image[i].g + 0.114 * original_image[i].b);
    }*/
    
    struct thread_cinza *args_cinza = malloc(thread_count * sizeof(struct thread_cinza));// = (struct thread_cinza *)malloc(sizeof(struct thread_cinza));

    for (thread = 0; thread < thread_count; thread++){
        args_cinza[thread].rank = thread;
        args_cinza[thread].final_gray_output = final_gray;
        args_cinza[thread].sharpened_image = sharpened_image;
        args_cinza[thread].width = width;
        args_cinza[thread].height = height;
        args_cinza[thread].thread_count = thread_count; 
        //printf("pronto pra criar threads\n");
        pthread_create(&thread_handles[thread], NULL,
        cinza, (void *)&args_cinza);
    }

    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }
    free(args_cinza);
    //printf("%s", (unsigned char)original_image[600][600]);

    // --------------------------------------------------------------------------
    // 3. ESCRITA DO ARQUIVO DE SAÍDA
    // --------------------------------------------------------------------------
    printf("Escrevendo imagem de saída em '%s'...\n", output_filename);
    FILE *outputFile = fopen(output_filename, "w");
    if (!outputFile) {
        perror("Erro ao criar o arquivo de saída");
        free(original_image); free(final_gray); free(blurred_image); free(sharpened_image);
        return 1;
    }

    fprintf(outputFile, "P3\n%d %d\n%d\n", width, height, max_val);
    for (int i = 0; i < width * height; i++) {
        fprintf(outputFile, "%d\n", final_gray[i]);
    }
    fclose(outputFile);

    // 4. LIBERAÇÃO DE MEMÓRIA
    free(original_image);
    free(final_gray);
    free(blurred_image);
    free(sharpened_image);
    
    free(thread_handles);

    printf("Filtro aplicado com sucesso!\n");

    return 0;
}