#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "symnmf.h"

/* Constants */
static const double EPSILON   = 1e-4;
static const int    MAX_ITER  = 300;
static char *goal;
static char *fileName;
static int   N_size;

/* Error printing helper */
static int error(){
    printf("An Error Has Occurred\n");
    return 1;
}

/* Vector (Coordinate) linked lists */

/* Create a new coordinate node. On allocation failure, delete the existing list. */
cord* new_cord(cord* curr_list, double val){
    cord* new_c = (cord*)malloc(sizeof(cord));
    if(new_c == NULL){
        delete_list_cord(curr_list);
        return NULL;
    }
    new_c->next = NULL;
    new_c->val  = val;
    return new_c;
}

/* Create a new vector node with an already-built coordinate list. */
vector* new_vector(vector* curr_list, cord* c){
    vector* v;
    if(c == NULL){
        delete_list_vector(curr_list, 1);
        return NULL;
    }
    v = (vector*)malloc(sizeof(vector));
    if(v == NULL){
        delete_list_cord(c);
        delete_list_vector(curr_list, 1);
        return NULL;
    }
    v->next  = NULL;
    v->cords = c;
    return v;
}

/* Recursively free coordinate list */
void delete_list_cord(cord *head){
    if(head != NULL){
        delete_list_cord(head->next);
        free(head);
    }
}

/* Recursively free vector list; if deleteCords, also free inner coordinate lists */
void delete_list_vector(vector *head, int deleteCords){
    if(head != NULL){
        if(deleteCords) delete_list_cord(head->cords);
        delete_list_vector(head->next, deleteCords);
        free(head);
    }
}

/* Squared Euclidean distance between two coordinate lists of equal length */
double squared_euclidean_distance(cord *a, cord *b){
    double res = 0.0;
    while(a != NULL){
        double diff = (a->val) - (b->val);
        res += diff * diff;
        a = a->next;
        b = b->next;
    }
    return res;
}

/* exp(-||vi - vj||^2 / 2) */
double calculate_exp_for_sym(vector* curr_v_i, vector* curr_v_j){
    double d2 = squared_euclidean_distance(curr_v_i->cords, curr_v_j->cords);
    return exp(-0.5 * d2);
}

/* Matrix helpers */

double** free_matrix(double **A, int R){
    int i;
    if(A != NULL){
        for(i = 0; i < R; ++i){
            if(A[i] != NULL) free(A[i]);
        }
        free(A);
    }
    return NULL;
}

double** create_matrix(int R, int C){
    int i;
    double **A = (double**)calloc(R, sizeof(double*));
    if(A == NULL) return NULL;
    for(i = 0; i < R; ++i){
        A[i] = (double*)calloc(C, sizeof(double));
        if(A[i] == NULL){
            return free_matrix(A, R);
        }
    }
    return A;
}

void print_matrix(double **A, int R, int C){
    int i, j;
    if(A == NULL) return;
    for(i = 0; i < R; ++i){
        for(j = 0; j < C; ++j){
            if(j != C - 1) printf("%.4f,", A[i][j]);
            else            printf("%.4f",   A[i][j]);
        }
        printf("\n");
    }
}

double** copy_matrix(double **A, int R, int C){
    int i, j;
    double **B;
    if(A == NULL) return NULL;
    B = create_matrix(R, C);
    if(B == NULL) return NULL;
    for(i = 0; i < R; ++i){
        for(j = 0; j < C; ++j){
            B[i][j] = A[i][j];
        }
    }
    return B;
}

/* C = A(R1xC1) * B(C1xC2) */
double** matrix_multiplication(double **A, double **B, int R1, int C1, int C2){
    int i, j, k;
    double **C;
    if(A == NULL || B == NULL) return NULL;
    C = create_matrix(R1, C2);
    if(C == NULL) return NULL;
    for(i = 0; i < R1; ++i){
        for(j = 0; j < C2; ++j){
            double sum = 0.0;
            for(k = 0; k < C1; ++k){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

/* B = A^T (C x R) */
double** matrix_create_Transpose(double **A, int R, int C){
    int i, j;
    double **B;
    if(A == NULL) return NULL;
    B = create_matrix(C, R);
    if(B == NULL) return NULL;
    for(i = 0; i < R; ++i){
        for(j = 0; j < C; ++j){
            B[j][i] = A[i][j];
        }
    }
    return B;
}

/* D = A - B */
double** matrix_subtraction(double **A, double **B, int R, int C){
    int i, j;
    double **D;
    if(A == NULL || B == NULL) return NULL;
    D = create_matrix(R, C);
    if(D == NULL) return NULL;
    for(i = 0; i < R; ++i){
        for(j = 0; j < C; ++j){
            D[i][j] = A[i][j] - B[i][j];
        }
    }
    return D;
}

/* In-place: A[i][i] <- (A[i][i])^(-1/2) on the diagonal of a square matrix */
void diagnol_matrix_negative_1_2(double **A, int N){
    int i;
    if(A == NULL) return;
    for(i = 0; i < N; ++i){
        /* Guard against division by zero: if diagonal is zero, leave as zero */
        if(A[i][i] > 0.0){
            A[i][i] = 1.0 / sqrt(A[i][i]);
        } else {
            A[i][i] = 0.0;
        }
    }
}

/* Frobenius norm squared of A */
static double squared_frobenius_norm(double **A, int R, int C){
    int i, j;
    double sum = 0.0;
    if(A == NULL) return -1.0;
    for(i = 0; i < R; ++i){
        for(j = 0; j < C; ++j){
            double v = A[i][j];
            sum += v * v;
        }
    }
    return sum;
}

/* ||A - B||_F^2 ; returns -1.0 on error */
double calculate_convergance(double **A, double **B, int R, int C){
    double **sub;
    double fn;
    if(A == NULL || B == NULL) return -1.0;
    sub = matrix_subtraction(A, B, R, C);
    if(sub == NULL) return -1.0;
    fn = squared_frobenius_norm(sub, R, C);
    free_matrix(sub, R);
    return fn;
}

/* IO and reading from command-line helpers */

/* returns 0 on invalid args */
static int read_arguments(int argc, char *argv[]){
    if(argc != 3){
        return 0;
    }
    goal = argv[1];
    if(strcmp(goal, "sym") && strcmp(goal, "ddg") && strcmp(goal, "norm")){
        return 0;
    }
    fileName = argv[2];
    return 1;
}

/* Read file; each line is one vector. */
static vector* readFile(){
    vector *head_vec = NULL, *curr_vec = NULL, *temp_v = NULL;
    cord   *head_cord = NULL, *curr_cord = NULL, *temp_c = NULL;
    double n;
    char   c;
    FILE *file = fopen(fileName, "r");
    if(file == NULL) return NULL;
    N_size = 0;

    while(fscanf(file, "%lf%c", &n, &c) == 2){
        temp_c = new_cord(head_cord, n);
        if(temp_c == NULL){
            delete_list_vector(head_vec, 1);
            fclose(file);
            return NULL;
        }
        if(head_cord == NULL) head_cord = temp_c;
        else                  curr_cord->next = temp_c;
        curr_cord = temp_c;

        if(c == '\n'){
            N_size += 1;
            temp_v = new_vector(head_vec, head_cord);
            if(temp_v == NULL){
                fclose(file);
                return NULL; /* inner lists already freed by new_vector on failure */
            }
            if(head_vec == NULL) head_vec = temp_v;
            else                 curr_vec->next = temp_v;
            curr_vec  = temp_v;
            head_cord = NULL;
            curr_cord = NULL;
        }
    }
    fclose(file);
    return head_vec;
}

/* Main SymNMF module */

/* Similarity matrix W with zeros on diagonal */
double** sym(vector* X, int N){
    int i, j;
    vector *vi = X, *vj;
    double **W;
    if(X == NULL) return NULL;
    W = create_matrix(N, N);
    if(W == NULL) return NULL;

    for(i = 0; i < N; ++i){
        vj = X;
        for(j = 0; j < N; ++j){
            if(i == j) W[i][j] = 0.0;
            else       W[i][j] = calculate_exp_for_sym(vi, vj);
            vj = vj->next;
        }
        vi = vi->next;
    }
    return W;
}

/* Diagonal degree matrix D from W (D_ii = sum_j W_ij) */
static double** ddg_logic(double **W, int N){
    int i, j;
    double **D;
    if(W == NULL) return NULL;
    D = create_matrix(N, N);
    if(D == NULL) return NULL;
    for(i = 0; i < N; ++i){
        double row_sum = 0.0;
        for(j = 0; j < N; ++j){
            row_sum += W[i][j];
        }
        D[i][i] = row_sum;
    }
    return D;
}

double** ddg(vector* X, int N){
    double **W = sym(X, N);
    double **D = ddg_logic(W, N);
    free_matrix(W, N);
    return D;
}

/* Normalized similarity W_norm = D^{-1/2} W D^{-1/2} */
double** norm(vector* X, int N){
    double **W = sym(X, N);
    double **D = ddg_logic(W, N);
    double **tmp, **Wn;

    diagnol_matrix_negative_1_2(D, N);
    tmp = matrix_multiplication(D, W, N, N, N);
    Wn  = matrix_multiplication(tmp, D, N, N, N);

    free_matrix(tmp, N);
    free_matrix(W, N);
    free_matrix(D, N);
    return Wn;
}

/* Multiplicative update rule for symmetric NMF
 * H_{t+1} = H_t * ((1 - beta) + beta * (W H_t) ./ (H_t H_t^T H_t))
 */
double** symnmf(double **H, double **W, int N, int k){
    int i, j, iter = 0;
    int ok = 1;
    const double beta = 0.5;
    const double tiny = 1e-12; /* denom guard */
    double **Ht[2], **HtT = NULL, **WH = NULL, **HtHtT = NULL, **tmp = NULL;
    double c;

    if(H == NULL || W == NULL) return NULL;

    Ht[0] = copy_matrix(H, N, k);
    Ht[1] = create_matrix(N, k);

    do{
        /* Numerator: W * H_t */
        WH   = matrix_multiplication(W, Ht[iter % 2], N, N, k);

        /* Denominator: H_t * (H_t^T * H_t) */
        HtT  = matrix_create_Transpose(Ht[iter % 2], N, k);   /* k x N */
        tmp  = matrix_multiplication(HtT, Ht[iter % 2], k, N, k);      /* k x k */
        HtHtT= matrix_multiplication(Ht[iter % 2], tmp, N, k, k);      /* N x k */

        ok = ok && (Ht[0] != NULL && Ht[1] != NULL &&
                    WH != NULL && HtT != NULL && tmp != NULL && HtHtT != NULL);

        if(ok){
            for(i = 0; i < N; ++i){
                for(j = 0; j < k; ++j){
                    double denom = HtHtT[i][j];
                    if (denom < tiny) denom = tiny; /* avoid divide-by-zero / blow-up */

                    Ht[(iter + 1) % 2][i][j] =
                        Ht[iter % 2][i][j] * ((1.0 - beta) + (beta * WH[i][j]) / denom);

                    if(Ht[(iter + 1) % 2][i][j] < 0.0){
                        Ht[(iter + 1) % 2][i][j] = 0.0; /* enforce non-negativity */
                    }
                }
            }
        }

        /* free temporaries for this iteration */
        free_matrix(tmp,   k);
        free_matrix(WH,    N);
        free_matrix(HtT,   k);
        free_matrix(HtHtT, N);

        iter++;
        c  = calculate_convergance(Ht[iter % 2], Ht[(iter - 1) % 2], N, k);
        ok = ok && (c >= 0.0); /* -1.0 means error */
    } while(ok && (c >= EPSILON) && (iter < MAX_ITER));

    /* keep the last valid H, delete the other */
    free_matrix(Ht[(iter + 1) % 2], N);
    return ok ? Ht[iter % 2] : NULL;
}


/* The main function for standalone work */

int main(int argc, char *argv[]){
    vector *allElements;
    double **res = NULL;

    if(!read_arguments(argc, argv)) return error();

    allElements = readFile();
    if(allElements == NULL) return error();

    if(strcmp(goal, "sym") == 0){
        res = sym(allElements, N_size);
    } else if(strcmp(goal, "ddg") == 0){
        res = ddg(allElements, N_size);
    } else if(strcmp(goal, "norm") == 0){
        res = norm(allElements, N_size);
    }

    if(res == NULL){
        delete_list_vector(allElements, 1);
        return error();
    }

    print_matrix(res, N_size, N_size);
    delete_list_vector(allElements, 1);
    free_matrix(res, N_size);
    return 0;
}