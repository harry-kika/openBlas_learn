//
// Created by canyu on 2021/2/6.
//

#ifndef X86CONV_TRANSFORMER_H
#define X86CONV_TRANSFORMER_H

#include <stdio.h>
#include <fstream>

class Transformer
{
public:

    Transformer(){};

    int load_param(int d_model_, int nhead_, int dim_feedforward_);
    void set(int d_model_, int nhead_, int dim_feedforward_, float* self_attn_q_weight_, float* self_attn_k_weight_,
                        float* self_attn_v_weight_, float* self_attn_q_bias_, float* self_attn_k_bias_,
                        float* self_attn_v_bias_, float* self_attn_out_weight_, float* self_attn_out_bias_,
                        float* norm1_weight_, float* norm1_bias_, float* linear1_weight_, float* linear1_bias_,
                        float* linear2_weight_, float* linear2_bias_, float* norm2_weight_, float* norm2_bias_);
    int load_model(FILE* &fp);
    int forward(float* src, int bs, int in_c, float* dest, float* mask);
    int transpose(float* src, int height, int width);
    int matrix_split(float* src, float* dest, int height, int width, int start, int len);
    int matrix_split_transpose(float* src, float* dest, int height, int width, int start, int len);
    int softmax(float* src, int height, int width);
    int layer_norm(float* src, int bs, int in_c, float* gamma, float* beta);
    int shortcut(float* src, float* dest, int height, int width);
    int position_embeding(float* src, int height, int width, float* mask);


private:
    int d_model;
    int nhead;
    int dim_feedforward;
    int head_dim;
    float scaling;

    float* self_attn_q_weight;
    float* self_attn_k_weight;
    float* self_attn_v_weight;
    float* self_attn_q_bias;
    float* self_attn_k_bias;
    float* self_attn_v_bias;
    float* self_attn_out_weight;
    float* self_attn_out_bias;
    float* norm1_weight;
    float* norm1_bias;
    float* linear1_weight;
    float* linear1_bias;
    float* linear2_weight;
    float* linear2_bias;
    float* norm2_weight;
    float* norm2_bias;


};

class Transformer_SVD
{
public:

    Transformer_SVD(){};

    int load_param(int d_model_, int nhead_, int dim_feedforward_);
    void set(int d_model_, int nhead_, int dim_feedforward_, float* self_attn_q_weight_, float* self_attn_k_weight_,
             float* self_attn_v_weight_, float* self_attn_q_bias_, float* self_attn_k_bias_,
             float* self_attn_v_bias_, float* self_attn_out_weight_, float* self_attn_out_bias_,
             float* norm1_weight_, float* norm1_bias_, float* linear1_svd0_weight_, float* linear1_svd1_weight_,
             float* linear1_svd1_bias_,float* linear2_svd0_weight_, float* linear2_svd1_weight_,
             float* linear2_svd1_bias_,float* norm2_weight_, float* norm2_bias_, int transformer_svd_dims_);
    int load_model(FILE* &fp);
    int forward(float* src, int bs, int in_c, float* dest, float* mask);
    int transpose(float* src, int height, int width);
    int matrix_split(float* src, float* dest, int height, int width, int start, int len);
    int matrix_split_transpose(float* src, float* dest, int height, int width, int start, int len);
    int softmax(float* src, int height, int width);
    int layer_norm(float* src, int bs, int in_c, float* gamma, float* beta);
    int shortcut(float* src, float* dest, int height, int width);
    int position_embeding(float* src, int height, int width, float* mask);


private:
    int d_model;
    int nhead;
    int dim_feedforward;
    int head_dim;
    float scaling;
    int transformer_svd_dims;

    float* self_attn_q_weight;
    float* self_attn_k_weight;
    float* self_attn_v_weight;
    float* self_attn_q_bias;
    float* self_attn_k_bias;
    float* self_attn_v_bias;
    float* self_attn_out_weight;
    float* self_attn_out_bias;
    float* norm1_weight;
    float* norm1_bias;
    float* linear1_svd0_weight;
    float* linear1_svd1_weight;
    float* linear1_svd1_bias;
    float* linear2_svd0_weight;
    float* linear2_svd1_weight;
    float* linear2_svd1_bias;
    float* norm2_weight;
    float* norm2_bias;


};

#endif //X86CONV_TRANSFORMER_H
