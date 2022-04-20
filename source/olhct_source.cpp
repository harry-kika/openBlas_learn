//#include <android/asset_manager_jni.h>
//#include <android/log.h>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <math.h>
#include "olhct.h"
#include "cblas.h"
#include "save_bit.h"
#include "decode.cpp"
#include <map>
#include <ctime>
#include <numeric>

using namespace std;


int OLHCT::load_param(int pre_d_model_, int d_model_1_, int d_model_2_ , int n_lm_layer_,int fc_svd_dims_, int em_svd_dims_, int transformer_svd_dims_,int classes_)
{
    pre_d_model = pre_d_model_;
    d_model_1 = d_model_1_;
    d_model_2 = d_model_2_;
    n_lm_layer = n_lm_layer_;
    fc_svd_dims = fc_svd_dims_;
    em_svd_dims = em_svd_dims_;
    transformer_svd_dims = transformer_svd_dims_;
    classes = classes_;

    return 1;
}

int OLHCT::load_model(AAsset *fp)
{
    //1
    pre_conv_layers_0_0_weight = new float[pre_d_model*d_model_1*3*1];
    pre_conv_layers_0_0_bias = new float[d_model_1];

    //4
    pre_conv_layers_1_0_weight = new float[d_model_1*d_model_2*3*1];
    pre_conv_layers_1_0_bias = new float[d_model_2];


    //2
    encoder_layers_0_self_attn_weight = new float[3*d_model_1*d_model_1]; //q,k,v
    // encoder_layers_0_self_attn_in_bias = new float[3*d_model_1] //q,k,v
    encoder_layers_0_self_attn_q_weight = new float[d_model_1*d_model_1];
    encoder_layers_0_self_attn_k_weight = new float[d_model_1*d_model_1];
    encoder_layers_0_self_attn_v_weight = new float[d_model_1*d_model_1];
    encoder_layers_0_self_attn_q_bias = new float[d_model_1];
    encoder_layers_0_self_attn_k_bias = new float[d_model_1];
    encoder_layers_0_self_attn_v_bias = new float[d_model_1];
    encoder_layers_0_self_attn_out_weight = new float[d_model_1*d_model_1];  //attn_map
    encoder_layers_0_self_attn_out_bias = new float[d_model_1]; //attn_map
    encoder_layers_0_self_attn_linear1_weight = new float[2*d_model_1*d_model_1];
    encoder_layers_0_self_attn_linear1_bias = new float[2*d_model_1];
    encoder_layers_0_self_attn_linear2_weight = new float[d_model_1*d_model_1*2];
    encoder_layers_0_self_attn_linear2_bias = new float[d_model_1];
    encoder_layers_0_self_attn_norm1_weight = new float[d_model_1];
    encoder_layers_0_self_attn_norm1_bias = new float[d_model_1];
    encoder_layers_0_self_attn_norm2_weight = new float[d_model_1];
    encoder_layers_0_self_attn_norm2_bias = new float[d_model_1];


    //5
    encoder_layers_1_self_attn_weight = new float[3*d_model_2*d_model_2]; //q,k,v
    // encoder_layers_1_self_attn_in_bias = new float[3*d_model_2] //q,k,v
    encoder_layers_1_self_attn_q_weight = new float[d_model_2*d_model_2];
    encoder_layers_1_self_attn_k_weight = new float[d_model_2*d_model_2];
    encoder_layers_1_self_attn_v_weight = new float[d_model_2*d_model_2];
    encoder_layers_1_self_attn_q_bias = new float[d_model_2];
    encoder_layers_1_self_attn_k_bias = new float[d_model_2];
    encoder_layers_1_self_attn_v_bias = new float[d_model_2];
    encoder_layers_1_self_attn_out_weight = new float[d_model_2*d_model_2];  //attn_map
    encoder_layers_1_self_attn_out_bias = new float[d_model_2]; //attn_map
    encoder_layers_1_self_attn_linear1_svd0_weight = new float[d_model_2*transformer_svd_dims];
    encoder_layers_1_self_attn_linear1_svd1_weight = new float[transformer_svd_dims*d_model_2*2];
    encoder_layers_1_self_attn_linear1_svd1_bias = new float[d_model_2*2];
    encoder_layers_1_self_attn_linear2_svd0_weight = new float[2*d_model_2*transformer_svd_dims];
    encoder_layers_1_self_attn_linear2_svd1_weight = new float[transformer_svd_dims*d_model_2];
    encoder_layers_1_self_attn_linear2_svd1_bias = new float[d_model_2];
    encoder_layers_1_self_attn_norm1_weight = new float[d_model_2];
    encoder_layers_1_self_attn_norm1_bias = new float[d_model_2];
    encoder_layers_1_self_attn_norm2_weight = new float[d_model_2];
    encoder_layers_1_self_attn_norm2_bias = new float[d_model_2];

    //1->2, 3->4
    norm_layers_0_weight = new float[d_model_1];
    norm_layers_0_bias = new float[d_model_1];
    //4->5, 6->7
    norm_layers_1_weight = new float[d_model_2];
    norm_layers_1_bias = new float[d_model_2];

    //3
    post_conv_layers_0_0_weight = new float[d_model_1*d_model_1*3*1];
    post_conv_layers_0_0_bias = new float[d_model_1];

    //6
    post_conv_layers_1_0_weight = new float[d_model_2*d_model_2*3*1];
    post_conv_layers_1_0_bias = new float[d_model_2];

    //share_classifier
    classifier_svd0_weight = new float[d_model_2*fc_svd_dims];
    classifier_svd1_weight = new float[(classes+1)*fc_svd_dims];
    classifier_svd1_bias = new float[(classes+1)];

    //lm_model
    //lm_1
    lm_encoder_layers_0_self_attn_weight = new float[3*d_model_2*d_model_2]; //q,k,v
    // lm_encoder_layers_0_self_attn_in_bias = new float[3*d_model_2] //q,k,v
    lm_encoder_layers_0_self_attn_q_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_0_self_attn_k_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_0_self_attn_v_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_0_self_attn_q_bias = new float[d_model_2];
    lm_encoder_layers_0_self_attn_k_bias = new float[d_model_2];
    lm_encoder_layers_0_self_attn_v_bias = new float[d_model_2];
    lm_encoder_layers_0_self_attn_out_weight = new float[d_model_2*d_model_2];  //attn_map
    lm_encoder_layers_0_self_attn_out_bias = new float[d_model_2]; //attn_map
    lm_encoder_layers_0_self_attn_linear1_svd0_weight = new float[transformer_svd_dims*d_model_2];
    lm_encoder_layers_0_self_attn_linear1_svd1_weight = new float[2*d_model_2*transformer_svd_dims];
    lm_encoder_layers_0_self_attn_linear1_svd1_bias = new float[2*d_model_2];
    lm_encoder_layers_0_self_attn_linear2_svd0_weight = new float[transformer_svd_dims*d_model_2*2];
    lm_encoder_layers_0_self_attn_linear2_svd1_weight = new float[d_model_2*transformer_svd_dims];
    lm_encoder_layers_0_self_attn_linear2_svd1_bias = new float[d_model_2];
    lm_encoder_layers_0_self_attn_norm1_weight = new float[d_model_2];
    lm_encoder_layers_0_self_attn_norm1_bias = new float[d_model_2];
    lm_encoder_layers_0_self_attn_norm2_weight = new float[d_model_2];
    lm_encoder_layers_0_self_attn_norm2_bias = new float[d_model_2];

    //lm_2
    lm_encoder_layers_1_self_attn_weight = new float[3*d_model_2*d_model_2]; //q,k,v
    // lm_encoder_layers_1_self_attn_in_bias = new float[3*d_model_2] //q,k,v
    lm_encoder_layers_1_self_attn_q_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_1_self_attn_k_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_1_self_attn_v_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_1_self_attn_q_bias = new float[d_model_2];
    lm_encoder_layers_1_self_attn_k_bias = new float[d_model_2];
    lm_encoder_layers_1_self_attn_v_bias = new float[d_model_2];
    lm_encoder_layers_1_self_attn_out_weight = new float[d_model_2*d_model_2];  //attn_map
    lm_encoder_layers_1_self_attn_out_bias = new float[d_model_2]; //attn_map
    lm_encoder_layers_1_self_attn_linear1_svd0_weight = new float[transformer_svd_dims*d_model_2];
    lm_encoder_layers_1_self_attn_linear1_svd1_weight = new float[2*d_model_2*transformer_svd_dims];
    lm_encoder_layers_1_self_attn_linear1_svd1_bias = new float[2*d_model_2];
    lm_encoder_layers_1_self_attn_linear2_svd0_weight = new float[transformer_svd_dims*d_model_2*2];
    lm_encoder_layers_1_self_attn_linear2_svd1_weight = new float[d_model_2*transformer_svd_dims];
    lm_encoder_layers_1_self_attn_linear2_svd1_bias = new float[d_model_2];
    lm_encoder_layers_1_self_attn_norm1_weight = new float[d_model_2];
    lm_encoder_layers_1_self_attn_norm1_bias = new float[d_model_2];
    lm_encoder_layers_1_self_attn_norm2_weight = new float[d_model_2];
    lm_encoder_layers_1_self_attn_norm2_bias = new float[d_model_2];

    //lm_3
    lm_encoder_layers_2_self_attn_weight = new float[3*d_model_2*d_model_2]; //q,k,v
    // lm_encoder_layers_2_self_attn_in_bias = new float[3*d_model_2] //q,k,v
    lm_encoder_layers_2_self_attn_q_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_2_self_attn_k_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_2_self_attn_v_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_2_self_attn_q_bias = new float[d_model_2];
    lm_encoder_layers_2_self_attn_k_bias = new float[d_model_2];
    lm_encoder_layers_2_self_attn_v_bias = new float[d_model_2];
    lm_encoder_layers_2_self_attn_out_weight = new float[d_model_2*d_model_2];  //attn_map
    lm_encoder_layers_2_self_attn_out_bias = new float[d_model_2]; //attn_map
    lm_encoder_layers_2_self_attn_linear1_svd0_weight = new float[transformer_svd_dims*d_model_2];
    lm_encoder_layers_2_self_attn_linear1_svd1_weight = new float[2*d_model_2*transformer_svd_dims];
    lm_encoder_layers_2_self_attn_linear1_svd1_bias = new float[2*d_model_2];
    lm_encoder_layers_2_self_attn_linear2_svd0_weight = new float[transformer_svd_dims*d_model_2*2];
    lm_encoder_layers_2_self_attn_linear2_svd1_weight = new float[d_model_2*transformer_svd_dims];
    lm_encoder_layers_2_self_attn_linear2_svd1_bias = new float[d_model_2];
    lm_encoder_layers_2_self_attn_norm1_weight = new float[d_model_2];
    lm_encoder_layers_2_self_attn_norm1_bias = new float[d_model_2];
    lm_encoder_layers_2_self_attn_norm2_weight = new float[d_model_2];
    lm_encoder_layers_2_self_attn_norm2_bias = new float[d_model_2];

    //lm_4
    lm_encoder_layers_3_self_attn_weight = new float[3*d_model_2*d_model_2]; //q,k,v
    // lm_encoder_layers_3_self_attn_in_bias = new float[3*d_model_2] //q,k,v
    lm_encoder_layers_3_self_attn_q_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_3_self_attn_k_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_3_self_attn_v_weight = new float[d_model_2*d_model_2];
    lm_encoder_layers_3_self_attn_q_bias = new float[d_model_2];
    lm_encoder_layers_3_self_attn_k_bias = new float[d_model_2];
    lm_encoder_layers_3_self_attn_v_bias = new float[d_model_2];
    lm_encoder_layers_3_self_attn_out_weight = new float[d_model_2*d_model_2];  //attn_map
    lm_encoder_layers_3_self_attn_out_bias = new float[d_model_2]; //attn_map
    lm_encoder_layers_3_self_attn_linear1_svd0_weight = new float[transformer_svd_dims*d_model_2];
    lm_encoder_layers_3_self_attn_linear1_svd1_weight = new float[2*d_model_2*transformer_svd_dims];
    lm_encoder_layers_3_self_attn_linear1_svd1_bias = new float[2*d_model_2];
    lm_encoder_layers_3_self_attn_linear2_svd0_weight = new float[transformer_svd_dims*d_model_2*2];
    lm_encoder_layers_3_self_attn_linear2_svd1_weight = new float[d_model_2*transformer_svd_dims];
    lm_encoder_layers_3_self_attn_linear2_svd1_bias = new float[d_model_2];
    lm_encoder_layers_3_self_attn_norm1_weight = new float[d_model_2];
    lm_encoder_layers_3_self_attn_norm1_bias = new float[d_model_2];
    lm_encoder_layers_3_self_attn_norm2_weight = new float[d_model_2];
    lm_encoder_layers_3_self_attn_norm2_bias = new float[d_model_2];

    //lm_embedding
    lm_embedding_svd0_weight = new float[(classes+1)*em_svd_dims];
    lm_embedding_svd1_weight = new float[em_svd_dims*d_model_2];;


    AAsset_read(fp, pre_conv_layers_0_0_weight, sizeof(float)*pre_d_model*d_model_1*3*1);
    AAsset_read(fp, pre_conv_layers_0_0_bias, sizeof(float)*d_model_1);

    unpack_cluster_blob(pre_conv_layers_1_0_weight, fp, d_model_1*d_model_2*3*1);
    AAsset_read(fp, pre_conv_layers_1_0_bias, sizeof(float)*d_model_2);


    AAsset_read(fp, encoder_layers_0_self_attn_q_weight, sizeof(float)*d_model_1*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_k_weight, sizeof(float)*d_model_1*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_v_weight, sizeof(float)*d_model_1*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_q_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_k_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_v_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_out_weight, sizeof(float)*d_model_1*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_out_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_linear1_weight, sizeof(float)*2*d_model_1*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_linear1_bias, sizeof(float)*2*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_linear2_weight, sizeof(float)*d_model_1*d_model_1*2);
    AAsset_read(fp, encoder_layers_0_self_attn_linear2_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_norm1_weight, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_norm1_bias,sizeof(float)* d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_norm2_weight, sizeof(float)*d_model_1);
    AAsset_read(fp, encoder_layers_0_self_attn_norm2_bias, sizeof(float)*d_model_1);

    unpack_cluster_blob(encoder_layers_1_self_attn_weight, fp, d_model_2*d_model_2*3);
    for(int i=0; i<d_model_2*d_model_2; i++){
        encoder_layers_1_self_attn_q_weight[i] = encoder_layers_1_self_attn_weight[i];
        encoder_layers_1_self_attn_k_weight[i] = encoder_layers_1_self_attn_weight[d_model_2*d_model_2 + i];
        encoder_layers_1_self_attn_v_weight[i] = encoder_layers_1_self_attn_weight[d_model_2*d_model_2*2 + i];
    }
    AAsset_read(fp, encoder_layers_1_self_attn_q_bias,sizeof(float)* d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_k_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_v_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(encoder_layers_1_self_attn_out_weight, fp, d_model_2*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_out_bias,sizeof(float)* d_model_2);
    unpack_cluster_blob(encoder_layers_1_self_attn_linear1_svd0_weight, fp, d_model_2*transformer_svd_dims);
    unpack_cluster_blob(encoder_layers_1_self_attn_linear1_svd1_weight, fp, 2*d_model_2*transformer_svd_dims);
    AAsset_read(fp, encoder_layers_1_self_attn_linear1_svd1_bias, sizeof(float)*2*d_model_2);
    unpack_cluster_blob(encoder_layers_1_self_attn_linear2_svd0_weight, fp, 2*d_model_2*transformer_svd_dims);
    unpack_cluster_blob(encoder_layers_1_self_attn_linear2_svd1_weight, fp, d_model_2*transformer_svd_dims);
    AAsset_read(fp, encoder_layers_1_self_attn_linear2_svd1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_norm1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_norm1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_norm2_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, encoder_layers_1_self_attn_norm2_bias, sizeof(float)*d_model_2);

    AAsset_read(fp, norm_layers_0_weight, sizeof(float)*d_model_1);
    AAsset_read(fp, norm_layers_0_bias, sizeof(float)*d_model_1);
    AAsset_read(fp, norm_layers_1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, norm_layers_1_bias, sizeof(float)*d_model_2);

    AAsset_read(fp, post_conv_layers_0_0_weight, sizeof(float)*d_model_1*d_model_1*3*1);
    AAsset_read(fp, post_conv_layers_0_0_bias, sizeof(float)*d_model_1);
    unpack_cluster_blob(post_conv_layers_1_0_weight, fp, d_model_2*d_model_2*3*1);
    AAsset_read(fp, post_conv_layers_1_0_bias, sizeof(float)*d_model_2);


    unpack_cluster_blob(classifier_svd0_weight, fp, fc_svd_dims*d_model_2);
    unpack_cluster_blob(classifier_svd1_weight, fp, (classes+1)*fc_svd_dims);
    AAsset_read(fp, classifier_svd1_bias, sizeof(float)*(classes+1));


    unpack_cluster_blob(lm_encoder_layers_0_self_attn_weight, fp, d_model_2*d_model_2*3);
    for(int i=0; i<d_model_2*d_model_2; i++){
        lm_encoder_layers_0_self_attn_q_weight[i] = lm_encoder_layers_0_self_attn_weight[i];
        lm_encoder_layers_0_self_attn_k_weight[i] = lm_encoder_layers_0_self_attn_weight[d_model_2*d_model_2 + i];
        lm_encoder_layers_0_self_attn_v_weight[i] = lm_encoder_layers_0_self_attn_weight[d_model_2*d_model_2*2 + i];
    }
    AAsset_read(fp, lm_encoder_layers_0_self_attn_q_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_k_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_v_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_0_self_attn_out_weight, fp, d_model_2*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_out_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_0_self_attn_linear1_svd0_weight, fp, d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_0_self_attn_linear1_svd1_weight, fp, 2*d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_linear1_svd1_bias, sizeof(float)*2*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_0_self_attn_linear2_svd0_weight, fp, 2*d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_0_self_attn_linear2_svd1_weight, fp, d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_linear2_svd1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_norm1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_norm1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_norm2_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_0_self_attn_norm2_bias, sizeof(float)*d_model_2);


    unpack_cluster_blob(lm_encoder_layers_1_self_attn_weight, fp, d_model_2*d_model_2*3);
    for(int i=0; i<d_model_2*d_model_2; i++){
        lm_encoder_layers_1_self_attn_q_weight[i] = lm_encoder_layers_1_self_attn_weight[i];
        lm_encoder_layers_1_self_attn_k_weight[i] = lm_encoder_layers_1_self_attn_weight[d_model_2*d_model_2 + i];
        lm_encoder_layers_1_self_attn_v_weight[i] = lm_encoder_layers_1_self_attn_weight[d_model_2*d_model_2*2 + i];
    }
    AAsset_read(fp, lm_encoder_layers_1_self_attn_q_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_k_bias,sizeof(float)* d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_v_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_1_self_attn_out_weight, fp, d_model_2*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_out_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_1_self_attn_linear1_svd0_weight, fp, d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_1_self_attn_linear1_svd1_weight, fp, 2*d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_linear1_svd1_bias, sizeof(float)*2*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_1_self_attn_linear2_svd0_weight, fp, 2*d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_1_self_attn_linear2_svd1_weight, fp, d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_linear2_svd1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_norm1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_norm1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_norm2_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_1_self_attn_norm2_bias, sizeof(float)*d_model_2);


    unpack_cluster_blob(lm_encoder_layers_2_self_attn_weight, fp, d_model_2*d_model_2*3);
    for(int i=0; i<d_model_2*d_model_2; i++){
        lm_encoder_layers_2_self_attn_q_weight[i] = lm_encoder_layers_2_self_attn_weight[i];
        lm_encoder_layers_2_self_attn_k_weight[i] = lm_encoder_layers_2_self_attn_weight[d_model_2*d_model_2 + i];
        lm_encoder_layers_2_self_attn_v_weight[i] = lm_encoder_layers_2_self_attn_weight[d_model_2*d_model_2*2 + i];
    }
    AAsset_read(fp, lm_encoder_layers_2_self_attn_q_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_k_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_v_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_2_self_attn_out_weight, fp, d_model_2*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_out_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_2_self_attn_linear1_svd0_weight, fp, d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_2_self_attn_linear1_svd1_weight, fp, 2*d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_linear1_svd1_bias, sizeof(float)*2*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_2_self_attn_linear2_svd0_weight, fp, 2*d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_2_self_attn_linear2_svd1_weight, fp, d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_linear2_svd1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_norm1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_norm1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_norm2_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_2_self_attn_norm2_bias, sizeof(float)*d_model_2);


    unpack_cluster_blob(lm_encoder_layers_3_self_attn_weight, fp, d_model_2*d_model_2*3);
    for(int i=0; i<d_model_2*d_model_2; i++){
        lm_encoder_layers_3_self_attn_q_weight[i] = lm_encoder_layers_3_self_attn_weight[i];
        lm_encoder_layers_3_self_attn_k_weight[i] = lm_encoder_layers_3_self_attn_weight[d_model_2*d_model_2 + i];
        lm_encoder_layers_3_self_attn_v_weight[i] = lm_encoder_layers_3_self_attn_weight[d_model_2*d_model_2*2 + i];
    }
    AAsset_read(fp, lm_encoder_layers_3_self_attn_q_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_k_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_v_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_3_self_attn_out_weight, fp, d_model_2*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_out_bias, sizeof(float)*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_3_self_attn_linear1_svd0_weight, fp, d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_3_self_attn_linear1_svd1_weight, fp, 2*d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_linear1_svd1_bias, sizeof(float)*2*d_model_2);
    unpack_cluster_blob(lm_encoder_layers_3_self_attn_linear2_svd0_weight, fp, 2*d_model_2*transformer_svd_dims);
    unpack_cluster_blob(lm_encoder_layers_3_self_attn_linear2_svd1_weight, fp, d_model_2*transformer_svd_dims);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_linear2_svd1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_norm1_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_norm1_bias, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_norm2_weight, sizeof(float)*d_model_2);
    AAsset_read(fp, lm_encoder_layers_3_self_attn_norm2_bias, sizeof(float)*d_model_2);

    unpack_cluster_blob(lm_embedding_svd0_weight, fp, (classes+1)*em_svd_dims);
    unpack_cluster_blob(lm_embedding_svd1_weight, fp, em_svd_dims*d_model_2);


    transpose(encoder_layers_0_self_attn_q_weight, d_model_1, d_model_1);
    transpose(encoder_layers_0_self_attn_k_weight, d_model_1, d_model_1);
    transpose(encoder_layers_0_self_attn_v_weight, d_model_1, d_model_1);
    transpose(encoder_layers_0_self_attn_linear1_weight, d_model_1*2, d_model_1);
    transpose(encoder_layers_0_self_attn_linear2_weight, d_model_1, d_model_1*2);

    transpose(encoder_layers_1_self_attn_q_weight, d_model_2, d_model_2);
    transpose(encoder_layers_1_self_attn_k_weight, d_model_2, d_model_2);
    transpose(encoder_layers_1_self_attn_v_weight, d_model_2, d_model_2);
    transpose(encoder_layers_1_self_attn_linear1_svd0_weight, transformer_svd_dims, d_model_2);
    transpose(encoder_layers_1_self_attn_linear1_svd1_weight, d_model_2*2, transformer_svd_dims);
    transpose(encoder_layers_1_self_attn_linear2_svd0_weight, transformer_svd_dims, d_model_2*2);
    transpose(encoder_layers_1_self_attn_linear2_svd1_weight, d_model_2, transformer_svd_dims);

    transpose(lm_encoder_layers_0_self_attn_q_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_0_self_attn_k_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_0_self_attn_v_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_0_self_attn_linear1_svd0_weight, transformer_svd_dims, d_model_2);
    transpose(lm_encoder_layers_0_self_attn_linear1_svd1_weight, d_model_2*2, transformer_svd_dims);
    transpose(lm_encoder_layers_0_self_attn_linear2_svd0_weight, transformer_svd_dims, d_model_2*2);
    transpose(lm_encoder_layers_0_self_attn_linear2_svd1_weight, d_model_2, transformer_svd_dims);

    transpose(lm_encoder_layers_1_self_attn_q_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_1_self_attn_k_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_1_self_attn_v_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_1_self_attn_linear1_svd0_weight, transformer_svd_dims, d_model_2);
    transpose(lm_encoder_layers_1_self_attn_linear1_svd1_weight, d_model_2*2, transformer_svd_dims);
    transpose(lm_encoder_layers_1_self_attn_linear2_svd0_weight, transformer_svd_dims, d_model_2*2);
    transpose(lm_encoder_layers_1_self_attn_linear2_svd1_weight, d_model_2, transformer_svd_dims);

    transpose(lm_encoder_layers_2_self_attn_q_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_2_self_attn_k_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_2_self_attn_v_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_2_self_attn_linear1_svd0_weight, transformer_svd_dims, d_model_2);
    transpose(lm_encoder_layers_2_self_attn_linear1_svd1_weight, d_model_2*2, transformer_svd_dims);
    transpose(lm_encoder_layers_2_self_attn_linear2_svd0_weight, transformer_svd_dims, d_model_2*2);
    transpose(lm_encoder_layers_2_self_attn_linear2_svd1_weight, d_model_2, transformer_svd_dims);

    transpose(lm_encoder_layers_3_self_attn_q_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_3_self_attn_k_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_3_self_attn_v_weight, d_model_2, d_model_2);
    transpose(lm_encoder_layers_3_self_attn_linear1_svd0_weight, transformer_svd_dims, d_model_2);
    transpose(lm_encoder_layers_3_self_attn_linear1_svd1_weight, d_model_2*2, transformer_svd_dims);
    transpose(lm_encoder_layers_3_self_attn_linear2_svd0_weight, transformer_svd_dims, d_model_2*2);
    transpose(lm_encoder_layers_3_self_attn_linear2_svd1_weight, d_model_2, transformer_svd_dims);

    transpose(classifier_svd0_weight, fc_svd_dims, 512);
//    transpose(classifier_svd1_weight, (classes+1), fc_svd_dims);

//    transpose(lm_embedding_svd0_weight, em_svd_dims, (classes+1));
//    transpose(lm_embedding_svd1_weight, d_model_2, em_svd_dims);



    return 1;
}

void OLHCT::set(){

    pre_conv_0_0.set(pre_d_model, d_model_1, 1, 3, 2, pre_conv_layers_0_0_weight, pre_conv_layers_0_0_bias);
    encoder_0.set(d_model_1, 8, d_model_1*2, encoder_layers_0_self_attn_q_weight, encoder_layers_0_self_attn_k_weight,
                  encoder_layers_0_self_attn_v_weight, encoder_layers_0_self_attn_q_bias, encoder_layers_0_self_attn_k_bias,
                  encoder_layers_0_self_attn_v_bias, encoder_layers_0_self_attn_out_weight, encoder_layers_0_self_attn_out_bias,
                  encoder_layers_0_self_attn_norm1_weight, encoder_layers_0_self_attn_norm1_bias, encoder_layers_0_self_attn_linear1_weight,
                  encoder_layers_0_self_attn_linear1_bias, encoder_layers_0_self_attn_linear2_weight, encoder_layers_0_self_attn_linear2_bias,
                  encoder_layers_0_self_attn_norm2_weight, encoder_layers_0_self_attn_norm2_bias);
    post_conv_0_0.set(d_model_1, d_model_1, 1, 3, 2, post_conv_layers_0_0_weight, post_conv_layers_0_0_bias);


    pre_conv_1_0.set(d_model_1, d_model_2, 1, 3, 2, pre_conv_layers_1_0_weight, pre_conv_layers_1_0_bias);
    encoder_1.set(d_model_2, 8, d_model_2*2, encoder_layers_1_self_attn_q_weight, encoder_layers_1_self_attn_k_weight,
                  encoder_layers_1_self_attn_v_weight, encoder_layers_1_self_attn_q_bias, encoder_layers_1_self_attn_k_bias,
                  encoder_layers_1_self_attn_v_bias, encoder_layers_1_self_attn_out_weight, encoder_layers_1_self_attn_out_bias,
                  encoder_layers_1_self_attn_norm1_weight, encoder_layers_1_self_attn_norm1_bias,
                  encoder_layers_1_self_attn_linear1_svd0_weight,encoder_layers_1_self_attn_linear1_svd1_weight,encoder_layers_1_self_attn_linear1_svd1_bias,
                  encoder_layers_1_self_attn_linear2_svd0_weight,encoder_layers_1_self_attn_linear2_svd1_weight,encoder_layers_1_self_attn_linear2_svd1_bias,
                  encoder_layers_1_self_attn_norm2_weight, encoder_layers_1_self_attn_norm2_bias, transformer_svd_dims);
    post_conv_1_0.set(d_model_2, d_model_2, 1, 3, 2, post_conv_layers_1_0_weight, post_conv_layers_1_0_bias);

    classifier_svd0.set(d_model_2, fc_svd_dims, classifier_svd0_weight, NULL,false);
    classifier_svd1.set(fc_svd_dims, (classes+1), classifier_svd1_weight, classifier_svd1_bias,true);

    lm_embedding_svd0.set((classes+1), em_svd_dims, lm_embedding_svd0_weight, NULL, false);
    lm_embedding_svd1.set(em_svd_dims, d_model_2, lm_embedding_svd1_weight, NULL, false);

    lm_encoder_0.set(d_model_2, 8, d_model_2*2, lm_encoder_layers_0_self_attn_q_weight, lm_encoder_layers_0_self_attn_k_weight, lm_encoder_layers_0_self_attn_v_weight,
                     lm_encoder_layers_0_self_attn_q_bias, lm_encoder_layers_0_self_attn_k_bias, lm_encoder_layers_0_self_attn_v_bias, lm_encoder_layers_0_self_attn_out_weight,
                     lm_encoder_layers_0_self_attn_out_bias, lm_encoder_layers_0_self_attn_norm1_weight, lm_encoder_layers_0_self_attn_norm1_bias,
                     lm_encoder_layers_0_self_attn_linear1_svd0_weight, lm_encoder_layers_0_self_attn_linear1_svd1_weight,lm_encoder_layers_0_self_attn_linear1_svd1_bias,
                     lm_encoder_layers_0_self_attn_linear2_svd0_weight, lm_encoder_layers_0_self_attn_linear2_svd1_weight,lm_encoder_layers_0_self_attn_linear2_svd1_bias,
                     lm_encoder_layers_0_self_attn_norm2_weight,lm_encoder_layers_0_self_attn_norm2_bias,transformer_svd_dims);
    lm_encoder_1.set(d_model_2, 8, d_model_2*2, lm_encoder_layers_1_self_attn_q_weight, lm_encoder_layers_1_self_attn_k_weight, lm_encoder_layers_1_self_attn_v_weight,
                     lm_encoder_layers_1_self_attn_q_bias, lm_encoder_layers_1_self_attn_k_bias, lm_encoder_layers_1_self_attn_v_bias, lm_encoder_layers_1_self_attn_out_weight,
                     lm_encoder_layers_1_self_attn_out_bias, lm_encoder_layers_1_self_attn_norm1_weight, lm_encoder_layers_1_self_attn_norm1_bias,
                     lm_encoder_layers_1_self_attn_linear1_svd0_weight, lm_encoder_layers_1_self_attn_linear1_svd1_weight,lm_encoder_layers_1_self_attn_linear1_svd1_bias,
                     lm_encoder_layers_1_self_attn_linear2_svd0_weight, lm_encoder_layers_1_self_attn_linear2_svd1_weight,lm_encoder_layers_1_self_attn_linear2_svd1_bias,
                     lm_encoder_layers_1_self_attn_norm2_weight,lm_encoder_layers_1_self_attn_norm2_bias,transformer_svd_dims);
    lm_encoder_2.set(d_model_2, 8, d_model_2*2, lm_encoder_layers_2_self_attn_q_weight, lm_encoder_layers_2_self_attn_k_weight, lm_encoder_layers_2_self_attn_v_weight,
                     lm_encoder_layers_2_self_attn_q_bias, lm_encoder_layers_2_self_attn_k_bias, lm_encoder_layers_2_self_attn_v_bias, lm_encoder_layers_2_self_attn_out_weight,
                     lm_encoder_layers_2_self_attn_out_bias, lm_encoder_layers_2_self_attn_norm1_weight, lm_encoder_layers_2_self_attn_norm1_bias,
                     lm_encoder_layers_2_self_attn_linear1_svd0_weight, lm_encoder_layers_2_self_attn_linear1_svd1_weight,lm_encoder_layers_2_self_attn_linear1_svd1_bias,
                     lm_encoder_layers_2_self_attn_linear2_svd0_weight, lm_encoder_layers_2_self_attn_linear2_svd1_weight,lm_encoder_layers_2_self_attn_linear2_svd1_bias,
                     lm_encoder_layers_2_self_attn_norm2_weight,lm_encoder_layers_2_self_attn_norm2_bias,transformer_svd_dims);
    lm_encoder_3.set(d_model_2, 8, d_model_2*2, lm_encoder_layers_3_self_attn_q_weight, lm_encoder_layers_3_self_attn_k_weight, lm_encoder_layers_3_self_attn_v_weight,
                     lm_encoder_layers_3_self_attn_q_bias, lm_encoder_layers_3_self_attn_k_bias, lm_encoder_layers_3_self_attn_v_bias, lm_encoder_layers_3_self_attn_out_weight,
                     lm_encoder_layers_3_self_attn_out_bias, lm_encoder_layers_3_self_attn_norm1_weight, lm_encoder_layers_3_self_attn_norm1_bias,
                     lm_encoder_layers_3_self_attn_linear1_svd0_weight, lm_encoder_layers_3_self_attn_linear1_svd1_weight,lm_encoder_layers_3_self_attn_linear1_svd1_bias,
                     lm_encoder_layers_3_self_attn_linear2_svd0_weight, lm_encoder_layers_3_self_attn_linear2_svd1_weight,lm_encoder_layers_3_self_attn_linear2_svd1_bias,
                     lm_encoder_layers_3_self_attn_norm2_weight,lm_encoder_layers_3_self_attn_norm2_bias,transformer_svd_dims);

}

vector< Results > OLHCT::forward(float* src, int bs)
{

    int points_num = bs;
    vector< Results > outputs;

    // ==> init key padding mask
    float* mask = new float[points_num]();
    for (int i = points_num - 128; i < points_num; i++) {
        mask[i] = 1.0;
    }

    float* pre_conv_0_0_output =  pre_conv_0_0.conv1d_forward(src, points_num, 1);
    transpose(pre_conv_0_0_output, d_model_1, points_num);
    layer_norm(pre_conv_0_0_output, points_num, d_model_1, norm_layers_0_weight, norm_layers_0_bias);
    // ==> mask downsample
    float* mask_0_0 = new float[points_num]();
    for (int i = 0; i < points_num / 2 * 2; i++) {
        mask_0_0[i] = mask[i * 2];
    }
    for (int i = points_num / 2 * 2; i < points_num; i++) {
        mask_0_0[i] = 1.0;
    }
    float* encoder_0_output = new float[points_num * d_model_1];
    encoder_0.forward(pre_conv_0_0_output, points_num, d_model_1, encoder_0_output, mask_0_0);
    float* post_conv_0_0_output = post_conv_0_0.conv1d_forward(encoder_0_output, points_num, 1);
    transpose(post_conv_0_0_output, d_model_1, points_num);
    layer_norm(post_conv_0_0_output, points_num, d_model_1, norm_layers_0_weight, norm_layers_0_bias);


    // ==> mask downsample
    float* mask_0_1 = new float[points_num]();
    for (int i = 0; i < points_num / 2 * 2; i++) {
        mask_0_1[i] = mask_0_0[i * 2];
    }
    for (int i = points_num / 2 * 2; i < points_num; i++) {
        mask_0_1[i] = 1.0;
    }

    float* pre_conv_1_0_output = pre_conv_1_0.conv1d_forward(post_conv_0_0_output, points_num, 1);
    transpose(pre_conv_1_0_output, d_model_2, points_num);
    layer_norm(pre_conv_1_0_output, points_num, d_model_2, norm_layers_1_weight, norm_layers_1_bias);
    // ==> mask downsample
    float* mask_1_0 = new float[points_num]();
    for (int i = 0; i < points_num / 2 * 2; i++) {
        mask_1_0[i] = mask_0_1[i * 2];
    }
    for (int i = points_num / 2 * 2; i < points_num; i++) {
        mask_1_0[i] = 1.0;
    }
    float* encoder_1_output = new float[points_num * d_model_2];
    encoder_1.forward(pre_conv_1_0_output, points_num, d_model_2, encoder_1_output, mask_1_0);

    float* post_conv_1_0_output = post_conv_1_0.conv1d_forward(encoder_1_output, points_num, 1);
    transpose(post_conv_1_0_output, d_model_2, points_num);
    layer_norm(post_conv_1_0_output, points_num, d_model_2, norm_layers_1_weight, norm_layers_1_bias);

    float* fc_net_output0 = classifier_svd0.ip_forward(post_conv_1_0_output, points_num);
    float* fc_net_output = classifier_svd1.ip_forward(fc_net_output0, points_num);
    softmax(fc_net_output, points_num, (classes+1));

    float* embedding_svd0_output = lm_embedding_svd0.ip_forward(fc_net_output, points_num);
    float* embedding_output = lm_embedding_svd1.ip_forward(embedding_svd0_output, points_num);


    // ==> mask downsample
    float* mask_1_1 = new float[points_num]();
    for (int i = 0; i < points_num; i++) {
        mask_1_1[i] = 0.0;
    }

    float* lm_encoder_0_output = new float[points_num * d_model_2];
    lm_encoder_0.forward(embedding_output, points_num, d_model_2, lm_encoder_0_output, mask_1_1);
    float* lm_encoder_1_output = new float[points_num * d_model_2];
    lm_encoder_1.forward(lm_encoder_0_output, points_num, d_model_2, lm_encoder_1_output, mask_1_1);
    float* lm_encoder_2_output = new float[points_num * d_model_2];
    lm_encoder_2.forward(lm_encoder_1_output, points_num, d_model_2, lm_encoder_2_output, mask_1_1);
    float* lm_encoder_3_output = new float[points_num * d_model_2];
    lm_encoder_3.forward(lm_encoder_2_output, points_num, d_model_2, lm_encoder_3_output, mask_1_1);

    float* fc_lm_output0 = classifier_svd0.ip_forward(lm_encoder_3_output, points_num);
    float* fc_lm_output = classifier_svd1.ip_forward(fc_lm_output0, points_num);

    softmax(fc_lm_output, points_num, (classes+1));

    vector<vector<pair<int,float>>> multiple_results;
    vector<float> confidence_array;

    multiple_results = CRUD(fc_lm_output, confidence_array, points_num, classes);


    vector<int> tmp_array;
    vector<vector<int>> top2_indexResults;

    for(int k = 0; k < multiple_results.size(); k++){
        for (int i = 0; i < multiple_results[k].size(); i++)
        {
            if (i == 0)
            {
                if (multiple_results[k][i].first != 0)
                {
                    tmp_array.push_back(multiple_results[k][i].first);
                }
            }
            else if (multiple_results[k][i].first != 0 && multiple_results[k][i].first != multiple_results[k][i - 1].first)
            {
                tmp_array.push_back(multiple_results[k][i].first);
            }

        }

        if(tmp_array.size()!=0){
            top2_indexResults.push_back(tmp_array);
        }

        tmp_array.clear();
    }


    Results results = Results(top2_indexResults, confidence_array);
    outputs.push_back(results);

    vector<vector<pair<int,float>>>().swap(multiple_results);
    vector<float>().swap(confidence_array);
    vector<int>().swap(tmp_array);
    vector<vector<int>>().swap(top2_indexResults);

    delete[] pre_conv_0_0_output;
    delete[] encoder_0_output;
    delete[] post_conv_0_0_output;
    delete[] pre_conv_1_0_output;
    delete[] encoder_1_output;
    delete[] post_conv_1_0_output;
    delete[] fc_net_output0;
    delete[] fc_net_output;
    delete[] embedding_svd0_output;
    delete[] embedding_output;

    delete[] lm_encoder_0_output;
    delete[] lm_encoder_1_output;
    delete[] lm_encoder_2_output;
    delete[] lm_encoder_3_output;
    delete[] fc_lm_output0;
    delete[] fc_lm_output;

    delete[] mask;
    delete[] mask_0_0;
    delete[] mask_0_1;
    delete[] mask_1_0;
    delete[] mask_1_1;

    results.release();


    return outputs;

}

int OLHCT::transpose(float *src, int height, int width) {
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

int OLHCT::softmax(float* src, int height, int width){

    for (int i = 0; i < height; i++)
    {
        float* ptr = src + i * width;
        float m = 0;
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


int OLHCT::layer_norm(float *src, int bs, int in_c, float* gamma_data, float* beta_data) {

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