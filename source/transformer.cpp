//
// Created by canyu on 2021/2/6.
//
//#include <android/log.h>
#include <iostream>
#include <cstring>
#include <math.h>
#include "transformer.h"
#include "cblas.h"
#include <float.h>


using namespace std;

int Transformer::load_param(int d_model_, int nhead_, int dim_feedforward_)
{
    d_model = d_model_;
    nhead = nhead_;
    dim_feedforward = dim_feedforward_;
    head_dim = d_model / nhead;
    scaling = 1.0 / sqrt((float)head_dim);

    return 0;
}

void Transformer::set(int d_model_, int nhead_, int dim_feedforward_, float* self_attn_q_weight_, float* self_attn_k_weight_,
                        float* self_attn_v_weight_, float* self_attn_q_bias_, float* self_attn_k_bias_,
                        float* self_attn_v_bias_, float* self_attn_out_weight_, float* self_attn_out_bias_,
                        float* norm1_weight_, float* norm1_bias_, float* linear1_weight_, float* linear1_bias_,
                        float* linear2_weight_, float* linear2_bias_, float* norm2_weight_, float* norm2_bias_)
{
    d_model = d_model_;
    nhead = nhead_;
    dim_feedforward = dim_feedforward_;
    head_dim = d_model / nhead;
    scaling = 1.0 / sqrt((float)head_dim);

    self_attn_q_weight = self_attn_q_weight_;
    self_attn_k_weight = self_attn_k_weight_;
    self_attn_v_weight = self_attn_v_weight_;
    self_attn_q_bias = self_attn_q_bias_;
    self_attn_k_bias = self_attn_k_bias_;
    self_attn_v_bias = self_attn_v_bias_;
    self_attn_out_weight = self_attn_out_weight_;
    self_attn_out_bias = self_attn_out_bias_;
    norm1_weight = norm1_weight_;
    norm1_bias = norm1_bias_;
    linear1_weight = linear1_weight_;
    linear1_bias = linear1_bias_;
    linear2_weight = linear2_weight_;
    linear2_bias = linear2_bias_;
    norm2_weight = norm2_weight_;
    norm2_bias = norm2_bias_;

}

int Transformer::load_model(FILE* &fp)
{

    self_attn_q_weight = new float[d_model * d_model];
    self_attn_k_weight = new float[d_model * d_model];
    self_attn_v_weight = new float[d_model * d_model];
    self_attn_q_bias = new float[d_model];
    self_attn_k_bias = new float[d_model];
    self_attn_v_bias = new float[d_model];
    self_attn_out_weight = new float[d_model * d_model];
    self_attn_out_bias = new float[d_model];
    norm1_weight = new float[d_model];
    norm1_bias = new float[d_model];
    linear1_weight = new float[d_model * dim_feedforward];
    linear1_bias = new float[dim_feedforward];
    linear2_weight = new float[dim_feedforward * d_model];
    linear2_bias = new float[d_model];
    norm2_weight = new float[d_model];
    norm2_bias = new float[d_model];

    fread(self_attn_q_weight, sizeof(float), d_model * d_model, fp);
    fread(self_attn_k_weight, sizeof(float), d_model * d_model, fp);
    fread(self_attn_v_weight, sizeof(float), d_model * d_model, fp);
    fread(self_attn_q_bias, sizeof(float), d_model, fp);
    fread(self_attn_k_bias, sizeof(float), d_model, fp);
    fread(self_attn_v_bias, sizeof(float), d_model, fp);
    fread(self_attn_out_weight, sizeof(float), d_model * d_model, fp);
    fread(self_attn_out_bias, sizeof(float), d_model, fp);
    fread(norm1_weight, sizeof(float), d_model, fp);
    fread(norm1_bias, sizeof(float), d_model, fp);
    fread(linear1_weight, sizeof(float), d_model * dim_feedforward, fp);
    fread(linear1_bias, sizeof(float), dim_feedforward, fp);
    fread(linear2_weight, sizeof(float), dim_feedforward * d_model, fp);
    fread(linear2_bias, sizeof(float), d_model, fp);
    fread(norm2_weight, sizeof(float), d_model, fp);
    fread(norm2_bias, sizeof(float), d_model, fp);

    transpose(self_attn_q_weight, d_model, d_model);
    transpose(self_attn_k_weight, d_model, d_model);
    transpose(self_attn_v_weight, d_model, d_model);
    transpose(linear1_weight, dim_feedforward, d_model);
    transpose(linear2_weight, d_model, dim_feedforward);
//    for(int i=0; i<d_model; i++){
//        cout << norm1_weight[i] << ", ";
//    }

    return 0;
}

int Transformer::forward(float* src, int bs, int in_c, float* dest, float* mask)
{

    if (in_c != d_model) return -1;

    float* pos_embed = new float[bs * in_c]();

    position_embeding(pos_embed, bs, in_c, mask);
    float* src_pos = new float[bs * in_c]();
    // position embeding
    for (int b_ = 0; b_ < bs; b_++) {
        for (int c = 0; c < in_c; c++) {
            int idx = b_ * in_c + c;
            src_pos[idx] = src[idx] + pos_embed[idx];
        }
    }
    delete[] pos_embed;

    float* q_vector = new float[bs * in_c];
    float* k_vector = new float[bs * in_c];
    float* v_vector = new float[bs * in_c];


    for (int i = 0; i < bs; i++) {
        memcpy(q_vector + i * in_c, self_attn_q_bias, in_c * sizeof(float));
        memcpy(k_vector + i * in_c, self_attn_k_bias, in_c * sizeof(float));
        memcpy(v_vector + i * in_c, self_attn_v_bias, in_c * sizeof(float));
    }

    // A * B = C
    const int M = bs; // 矩阵A的行，矩阵C的行
    const int N = in_c; // 矩阵B的列，矩阵C的列
    const int K = in_c; // 矩阵A的列，矩阵B的行
    const enum CBLAS_ORDER Order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
    const float alpha = 1;
    const float beta = 1;
    const int lda = K;//A的列
    const int ldb = N;//B的列
    const int ldc = N;//C的列
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, src_pos, lda, self_attn_q_weight, ldb, beta, q_vector, ldc);
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, src_pos, lda, self_attn_k_weight, ldb, beta, k_vector, ldc);
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, src, lda, self_attn_v_weight, ldb, beta, v_vector, ldc);

    // Q
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < in_c; j++) {
            q_vector[i * in_c + j] *= scaling;
        }
    }


    // 对于8个head，每个head单独计算
    int size_per_head = bs * head_dim;
    float* q1 = new float[size_per_head]();
    float* k1 = new float[size_per_head]();
    float* v1 = new float[size_per_head]();
    float* attn_output_weight = new float[bs * bs]();
    float* attn_output = new float[bs * in_c]();
    // head 0
    matrix_split(q_vector, q1, bs, in_c, 0, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 0, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 0, head_dim);

    float* attn_output_ptr = attn_output;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 1
    matrix_split(q_vector, q1, bs, in_c, 1 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 1 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 1 * head_dim, head_dim);

    attn_output_ptr = attn_output + 1 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 2
    matrix_split(q_vector, q1, bs, in_c, 2 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 2 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 2 * head_dim, head_dim);

    attn_output_ptr = attn_output + 2 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 3
    matrix_split(q_vector, q1, bs, in_c, 3 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 3 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 3 * head_dim, head_dim);

    attn_output_ptr = attn_output + 3 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 4
    matrix_split(q_vector, q1, bs, in_c, 4 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 4 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 4 * head_dim, head_dim);

    attn_output_ptr = attn_output + 4 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 5
    matrix_split(q_vector, q1, bs, in_c, 5 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 5 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 5 * head_dim, head_dim);

    attn_output_ptr = attn_output + 5 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 6
    matrix_split(q_vector, q1, bs, in_c, 6 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 6 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 6 * head_dim, head_dim);

    attn_output_ptr = attn_output + 6 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);

    // head 7
    matrix_split(q_vector, q1, bs, in_c, 7 * head_dim, head_dim);
    matrix_split_transpose(k_vector, k1, bs, in_c, 7 * head_dim, head_dim);
    matrix_split(v_vector, v1, bs, in_c, 7 * head_dim, head_dim);

    attn_output_ptr = attn_output + 7 * size_per_head;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, bs, head_dim, 1.0, q1, head_dim, k1, bs, 0.0, attn_output_weight, bs);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
            if (abs(mask[j] - 1.0) < 0.001) { // equal to 1
                attn_output_weight[i * bs + j] = -FLT_MAX;
            }
        }
    }
    softmax(attn_output_weight, bs, bs);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, head_dim, bs, 1.0, attn_output_weight, bs, v1, head_dim, 0.0, attn_output_ptr, head_dim);
    //
    float* attn_output_trans = new float[bs * in_c]();
    for (int c = 0; c < nhead; c++) {
        for (int h = 0; h < bs; h++) {
            for (int w = 0; w < head_dim; w++) {
                int idx = c * size_per_head + h * head_dim + w;
                attn_output_trans[h * head_dim * nhead + c * head_dim + w] = attn_output[idx];
            }
        }
    }

    float* self_attn_output = new float[bs * in_c];

    for (int i = 0; i < bs; i++) {
        memcpy(self_attn_output + i * in_c, self_attn_out_bias, in_c * sizeof(float));
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bs, in_c, in_c, 1.0, attn_output_trans, in_c, self_attn_out_weight, in_c, 1.0, self_attn_output, in_c);

    shortcut(src, self_attn_output, bs, in_c);

    // layer norm 1
    layer_norm(self_attn_output, bs, in_c, norm1_weight, norm1_bias);

    // linear 1 + relu + linear 2
    float* linear1_output = new float[bs * dim_feedforward]();
    for (int i = 0; i < bs; i++) {
        memcpy(linear1_output + i * dim_feedforward, linear1_bias, dim_feedforward * sizeof(float));
    }
    //    transpose(linear1_weight, dim_feedforward, d_model);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, dim_feedforward, d_model, 1.0, self_attn_output, d_model, linear1_weight, dim_feedforward, 1.0, linear1_output, dim_feedforward);
    // relu
    for (int i = 0; i < bs * dim_feedforward; i++) {
        linear1_output[i] = linear1_output[i] > 0 ? linear1_output[i] : 0;
    }
    //linear 2
    for (int i = 0; i < bs; i++) {
        memcpy(dest + i * d_model, linear2_bias, d_model * sizeof(float));
    }

    //    transpose(linear2_weight, d_model, dim_feedforward);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bs, d_model, dim_feedforward, 1.0, linear1_output, dim_feedforward, linear2_weight, d_model, 1.0, dest, d_model);

    shortcut(self_attn_output, dest, bs, in_c);

    layer_norm(dest, bs, in_c, norm2_weight, norm2_bias);



    delete[] q_vector;
    delete[] k_vector;
    delete[] v_vector;
    delete[] q1;
    delete[] k1;
    delete[] v1;
    delete[] attn_output_weight;
    delete[] attn_output;
    delete[] attn_output_trans;
    delete[] self_attn_output;
    delete[] linear1_output;
    delete[] src_pos;


    return 0;
}

int Transformer::matrix_split(float *src, float *dest, int height, int width, int start, int len) {

    for(int i=0; i<height; i++){
        int jj = 0;
        for(int j=start; j<start+len; j++){
            dest[i*len + jj] = src[i*width + j];
            jj++;
        }
    }
    return 0;

}

int Transformer::matrix_split_transpose(float *src, float *dest, int height, int width, int start, int len) {

    for(int i=0; i<height; i++){
        int jj = 0;
        for(int j=start; j<start+len; j++){
            dest[jj * height + i] = src[i*width + j];
            jj++;
        }
    }
    return 0;

}

int Transformer::transpose(float *src, int height, int width) {
    // convert A : a*b -> b*a
    float* tmp = new float[width * height];
    for(int i=0; i<width; i++){
        for(int j=0; j<height; j++){
            tmp[i * height + j] = src[j * width + i];
        }
    }
    memcpy(src, tmp, height * width *sizeof(float));
    delete[] tmp;

    return 0;
}

int Transformer::softmax(float* src, int height, int width){

    for (int i = 0; i < height; i++)
    {
        float* ptr = src + i * width;
        float m = -__FLT_MAX__;
        for (int j = 0; j < width; j++)
        {
            m = std::max(m, ptr[j]);
        }

        float s = 0.f;
        for (int j = 0; j < width; j++)
        {
            ptr[j] = static_cast<float>(exp(ptr[j] - m));
            s += ptr[j];
        }

        for (int j = 0; j < width; j++)
        {
            ptr[j] /= s;
        }
    }

    return 0;
}

int Transformer::shortcut(float *src, float *dest, int height, int width) {
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            dest[i * width + j] += src[i * width + j];
        }
    }
    return 0;
}

int Transformer::layer_norm(float *src, int bs, int in_c, float* gamma_data, float* beta_data) {

    // src : in_c * bs, 对于每个通道c，有一个weight跟bias

    // mean and var
    float sum = 0.f;
    float eps = 0.00001f;
    float* mean = new float[bs]();
    

    for (int q = 0; q < bs; q++)
    {
        const float* ptr = src + q * in_c;
        for (int i = 0; i < in_c; i++)
        {
            mean[q] += ptr[i];
        }
    }
    
    for(int i=0; i<bs; i++){
        mean[i] /= in_c;
    }
    float* var = new float[bs]();
    for (int q = 0; q < bs; q++)
    {
        const float* ptr = src + q * in_c;
        for (int i = 0; i < in_c; i++)
        {
            float tmp = ptr[i] - mean[q];
            var[q] += tmp * tmp;
        }
    }
    for(int i=0; i<bs; i++){
        var[i] /= in_c;
    }

//#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < bs; q++)
    {
        float* ptr = src + q * in_c;
        // cout<<"src: " << src[1] <<endl;
        for(int i=0; i<in_c; i++) {
            float gamma = gamma_data[i];
            float beta = beta_data[i];
            
            float a = static_cast<float>(gamma / sqrt(var[q] + eps));
            float b = -mean[q] * a + beta;
            ptr[i] = ptr[i] * a + b;
            
        }
    }
    delete[] mean;
    return 0;
}

int Transformer::position_embeding(float* src, int height, int width, float* mask) {

    int tempreture = 10000;
    int cumsum = -1;
    for (int i = 0; i < height; i++) {
        if (abs(mask[i] - 0) < 0.0001) {
            cumsum++;
        }
        for (int j = 0; j < width / 2; j++) {
            float dim_t = static_cast<float>(2 * j / (float)width);
            //            float cos_y = static_cast<float>((2*j)/(float)width);
            //float tmp1 = sin((i + 1) / pow(tempreture, dim_t));
            src[i * width + 2 * j] = sin((cumsum + 1) / pow(tempreture, dim_t));
            //float tmp2 = cos((i + 1) / pow(tempreture, dim_t));
            src[i * width + 2 * j + 1] = cos((cumsum + 1) / pow(tempreture, dim_t));
        }
        for (int j = width / 2 * 2; j < width; j++) {
            float sin_y = static_cast<float>(j / (float)width);
            src[i * width + j] = sin((cumsum + 1) / pow(tempreture, sin_y));
        }
    }
    return 0;

}