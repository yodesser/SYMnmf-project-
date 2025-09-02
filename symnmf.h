#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Raw datapoint representation (linked lists) */
typedef struct cord {
    double val;
    struct cord *next;
} cord;

typedef struct vector {
    struct vector *next;
    struct cord *cords;
} vector;

/* Constructors / destructors */
cord*   new_cord(cord* curr_list, double val);
vector* new_vector(vector* curr_list, cord* c);
void    delete_list_cord(cord *head);
void    delete_list_vector(vector *head, int deleteCords);

/* Utilities for datapoints */
double  squared_euclidean_distance(cord *a, cord *b);
double  calculate_exp_for_sym(vector* curr_v_i, vector* curr_v_j);

/* Matrix helpers */
double** free_matrix(double **A, int R);
double** create_matrix(int R, int C);
void     print_matrix(double **A, int R, int C);
double** copy_matrix(double **A, int R, int C);
double** matrix_multiplication(double **A, double **B, int R1, int C1, int C2);
double** matrix_create_Transpose(double **A, int R, int C);
double** matrix_subtraction(double **A, double **B, int R, int C);
void     diagnol_matrix_negative_1_2(double **A, int N);
double   calculate_convergance(double **A, double **B, int R, int C);

/* SymNMF-relateed routines */
double** sym(vector* X, int N);
double** ddg(vector* X, int N);
double** norm(vector* X, int N);
double** symnmf(double** H, double** W, int N, int k);

#endif /* SYMNMF_H_ */