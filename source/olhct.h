//
// Created by canyu on 2021/2/6.
//

#ifndef X86CONV_TRANSFORMER_H
#define X86CONV_TRANSFORMER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include "cblas.h"
#include <cstring>
#include "AAsset.h"

using namespace std;

class Results{
public:
	Results(vector<vector<int>> lm_outputs, vector<float> lm_confidence) {
		this->m_lm_outputs= lm_outputs;
		this->m_lm_confidence= lm_confidence;

	}
	void release(){
		vector<vector<int>>().swap(m_lm_outputs);
		vector<float>().swap(m_lm_confidence);
	}
	vector<vector<int>> m_lm_outputs;
	vector<float> m_lm_confidence;
};

class bn{
	
public:
	bn(int num_output_ , float * weight_ , int & weight_offset , bool scaleOrNot = false, float eps_ = 0.00001){
		num_output = num_output_;
		eps = eps_;
		scale = scaleOrNot;
		if(scale){
      		weight = weight_ + weight_offset;
      		weight_offset += num_output;
      		bias = weight_ + weight_offset;
      		weight_offset += num_output;
		}
		mean = weight_ + weight_offset;
		weight_offset += num_output;
		var = weight_ + weight_offset;
		weight_offset += num_output;
	}
	bn(){}
	
	void set(int num_output_ , float * weight_ , int & weight_offset , bool scaleOrNot = false, float eps_ = 0.00001)
	{
		num_output = num_output_;
		eps = eps_;
		scale = scaleOrNot;
		if(scale){
      		weight = weight_ + weight_offset;
      		weight_offset += num_output;
      		bias = weight_ + weight_offset;
      		weight_offset += num_output;
		}
		mean = weight_ + weight_offset;
		weight_offset += num_output;
		var = weight_ + weight_offset;
		weight_offset += num_output;
	}
	
	void set(FILE* &fp)
	{
		eps = 0.00001f;
		fread(&num_output, sizeof(int), 1, fp);
		fread(&scale, sizeof(int), 1, fp);

		weight = new float[num_output]();
		bias = new float[num_output]();
		mean = new float[num_output]();
		var = new float[num_output]();


		

		if(scale){
		
      		fread(weight, sizeof(float), num_output, fp);
      		fread(bias, sizeof(float), num_output, fp); 	
		}

		fread(mean, sizeof(float), num_output, fp);
		fread(var, sizeof(float), num_output, fp);

	}


	float * bn_forward (float * input_feature_map , int height_input , int width_input){	//compute 1 pic
		int length = width_input * height_input;
		float* output_feature_map = new float[num_output * length];
		for(int  i = 0 ; i < num_output * length ; i++)
		{	
			output_feature_map[i] = ( input_feature_map[i] - mean[i/length] ) / sqrt( var[i/length] + eps) ;
			if(scale){
				output_feature_map[i] = output_feature_map[i] * weight[i/length] + bias[i/length];
			}
		}
		delete input_feature_map;
		return output_feature_map;
	}

private:
	int num_output;
	float * weight;
	float * bias;
	float * mean;
	float * var;
	float eps;
	bool scale;

};

class conv1d{
public:
  
  //构造函数
  conv1d(int num_input_, int num_output_, int pad_w_, int kernel_w_,
 		int stride_w_,float* weight_, float* bias_, 
		bool with_bias_ = true, bool is_depthwise_= false, int channel = 1){

		num_input = num_input_;
		num_output = num_output_;

		pad_w = pad_w_;
		kernel_w = kernel_w_;
		stride_w = stride_w_;

		weight = weight_;

		with_bias = with_bias_;
		if (with_bias){
		bias = bias_;
		}
        is_depthwise = is_depthwise_;

	}


  conv1d(){};

  //层初始化
  void set(int num_input_, int num_output_, int pad_w_, int kernel_w_,
 		int stride_w_,float* weight_, float* bias_, 
		bool with_bias_ = true, bool is_depthwise_= false, int channel = 1){

		num_input = num_input_;
		num_output = num_output_;

		pad_w = pad_w_;
		pad_h = 0;
		kernel_w = kernel_w_;
		kernel_h = 1;
		stride_w = stride_w_;
		stride_h = 0;
		weight = weight_;
		with_bias = with_bias_;
		if (with_bias){
		bias = bias_;
		}
        is_depthwise = is_depthwise_;
	}


  //前向计算
  float* conv1d_forward(float* input_feature_map,
                        int& width_input, bool relu = false){

		int width_output = (width_input + pad_w * 2 - kernel_w) / stride_w +1;
//	  	__android_log_print(ANDROID_LOG_DEBUG, "olhct", "%d ", width_input);
		int output_length = width_output * num_output;
		float* output_feature_map = new float[output_length];
		//M: A's row, C's row
		//N: B's col, C's col
		//K: A's col, B's row
		const int M = num_output;
		const int N = width_output;
		const int K = kernel_w * num_input;

		if(is_depthwise){

			float* one_feature_map = new float[1 * width_input];
			float* processed_one_feature_map = new float[width_output * kernel_w];
			float* processed_one_weight = new float[1 * kernel_w];
			float* one_output_feature = new float[1 * width_output];
			const int a = 1;
			const int b = width_output;
			const int c = kernel_w;

			for(int j=0; j< num_output; j++){
				for(int i=0; i<width_input; i++){
				one_feature_map[i] = input_feature_map[j*width_input + i];
				}
				for (int c = 0; c < kernel_w; ++c) 
				{
				int w_offset = c % kernel_w;
				int c_im = c / kernel_w;
				for (int w = 0; w < width_output; ++w) 
				{
					int w_pad = w * stride_w - pad_w + w_offset;
					if (w_pad >= 0 && w_pad < width_input)
					processed_one_feature_map[c * width_output + w] = one_feature_map[c_im * width_input + w_pad];
					else
					processed_one_feature_map[c * width_output + w] = 0;
				}	
				}
				for(int i=0; i<kernel_w; i++){
					processed_one_weight[i] = weight[j*kernel_w + i];
				}

				CBLAS_ORDER Order = CblasRowMajor;
				CBLAS_TRANSPOSE TransA = CblasNoTrans;
				CBLAS_TRANSPOSE TransB = CblasNoTrans;
				cblas_sgemm(Order, TransA, TransB, a, b, c,
					1, processed_one_weight, c, processed_one_feature_map, b,
					0, one_output_feature, b);


				for(int i=0; i<width_output;i++){
					output_feature_map[width_output*j+i] = one_output_feature[i];
				}
			}
			
			delete[] processed_one_weight;
			delete[] processed_one_feature_map;
			delete[] one_feature_map;
			delete[] one_output_feature;
		}
		else{
			float* processed_input_feature_map = new float[K * N];
            float* tmp_transpose = new float[width_input * num_input];

            for(int i=0; i<num_input; i++){
                for(int j=0; j<width_input; j++){
                    tmp_transpose[i * width_input + j] = input_feature_map[j * num_input + i];
                }
            }
            memcpy(input_feature_map, tmp_transpose, width_input * num_input *sizeof(float));
            delete[] tmp_transpose;

			//im2col
			for (int c = 0; c < K; ++c)
			{
				int w_offset = c % kernel_w;
				int c_im = c / kernel_w;
				for (int w = 0; w < width_output; ++w)
				{
					int w_pad = w * stride_w - pad_w + w_offset;
					if (w_pad >= 0 && w_pad < width_input)
					processed_input_feature_map[c * width_output + w] = input_feature_map[c_im * width_input + w_pad];
					else
					processed_input_feature_map[c * width_output + w] = 0;
				}
			}

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;

            cblas_sgemm(Order, TransA, TransB, M, N, K,
                        1, weight, K, processed_input_feature_map, N,
                        0, output_feature_map, N);

			delete[] processed_input_feature_map;
		}

		if(relu){
			for(int output_index = 0; output_index < output_length; output_index++){
				if(with_bias){
				output_feature_map[output_index]+=
					bias[output_index / width_output];
				}
				output_feature_map[output_index] = output_feature_map[output_index] >= 0?
										output_feature_map[output_index] : 0;
				//relu6
//				output_feature_map[output_index] = output_feature_map[output_index] <= 6?
//										output_feature_map[output_index] : 6;
			}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
		}
		else{
			if(with_bias){
			for(int output_index = 0; output_index < output_length; output_index++){
				output_feature_map[output_index] += bias[output_index / width_output];
			}
			}
		}
		
		width_input = width_output;
		return output_feature_map;
	}

private:
  int num_input;
  int num_output;
  int pad_w;
  int pad_h;
  int kernel_w;
  int kernel_h;
  int stride_w;
  int stride_h;
  float* weight;
  float* bias;
  bool with_bias;
  bool is_depthwise;

};


class conv{
public:
  int weight_output;
  int height_output;
  //构造函数
  conv(int num_input_, int num_output_,
    int pad_h_, int pad_w_, int kernel_h_, int kernel_w_,
    int stride_h_, int stride_w_,
    float* weight_, float* bias_, bool with_bias_ = true, bool is_depthwise_= false, int channel = 1){


		num_input = num_input_;
		num_output = num_output_;

		pad_h = pad_h_;
		pad_w = pad_w_;
		kernel_h = kernel_h_;
		kernel_w = kernel_w_;
		stride_h = stride_h_;
		stride_w = stride_w_;

		weight = weight_;

		with_bias = with_bias_;
		if (with_bias){
		bias = bias_;
		}
        is_depthwise = is_depthwise_;

	}


  conv(){};

  //层初始化
  void set(int num_input_, int num_output_,
    int pad_h_, int pad_w_, int kernel_h_, int kernel_w_,
    int stride_h_, int stride_w_,
    float* weight_, float* bias_, bool with_bias_ = true, bool is_depthwise_= false, int channel = 1){

		num_input = num_input_;
		num_output = num_output_;

		pad_h = pad_h_;
		pad_w = pad_w_;
		kernel_h = kernel_h_;
		kernel_w = kernel_w_;
		stride_h = stride_h_;
		stride_w = stride_w_;

		weight = weight_;

		with_bias = with_bias_;
		if (with_bias){
		bias = bias_;
		}
        is_depthwise = is_depthwise_;
	}


  //前向计算
  float* conv_forward(float* input_feature_map, int& height_input,
                        int& width_input, bool relu = false){
		int height_output = (height_input + pad_h * 2 - kernel_h) / stride_h +1;
		int width_output = (width_input + pad_w * 2 - kernel_w) / stride_w +1;
		int output_length = height_output * width_output * num_output;
		float* output_feature_map = new float[output_length];
		//M: A's row, C's row
		//N: B's col, C's col
		//K: A's col, B's row
		const int M = num_output;
		const int N = height_output * width_output;
		const int K = kernel_h * kernel_w * num_input;
		float* processed_input_feature_map = new float[K * N];
		//im2col
		for (int c = 0; c < K; ++c) 
		{
			int w_offset = c % kernel_w;
			int h_offset = (c / kernel_w) % kernel_h;
			int c_im = c / kernel_h / kernel_w;
			for (int h = 0; h < height_output; ++h) 
			{
			for (int w = 0; w < width_output; ++w) 
			{
				int h_pad = h * stride_h - pad_h + h_offset;
				int w_pad = w * stride_w - pad_w + w_offset;
          
				if (h_pad >= 0 && h_pad < height_input && w_pad >= 0 && w_pad < width_input)
				processed_input_feature_map[(c * height_output + h) * width_output + w] = input_feature_map[(c_im * height_input + h_pad) * width_input + w_pad];
				else
				processed_input_feature_map[(c * height_output + h) * width_output + w] = 0;
			}
			}
		}
        float* processed_weight = new float[K * M];
        //depthwise_process (1d_conv)
        for (int tmp1 = 0; tmp1 < num_input; ++tmp1) 
		{
			float a = weight[tmp1*kernel_w + 0];
            float b = weight[tmp1*kernel_w + 1];
            float c = weight[tmp1*kernel_w + 2];
            
            for (int tmp2 = 0; tmp2 < num_output; ++tmp2) {
            processed_weight[tmp1*kernel_w*num_output + tmp2*kernel_w + 0] = a;
            processed_weight[tmp1*kernel_w*num_output + tmp2*kernel_w + 1] = b;
            processed_weight[tmp1*kernel_w*num_output + tmp2*kernel_w + 2] = c;
            }
            
            
		}
		const  CBLAS_ORDER Order = CblasRowMajor;
		const  CBLAS_TRANSPOSE TransA = CblasNoTrans;
		const  CBLAS_TRANSPOSE TransB = CblasNoTrans;
		/*cblas_dgemm(Order, TransA, TransB, M, N, K,
					1, weight, K, processed_input_feature_map, N,
					0, output_feature_map, N);*/
        if(is_depthwise){
            cblas_sgemm(Order, TransA, TransB, M, N, K,
					1, processed_weight, K, processed_input_feature_map, N,
					0, output_feature_map, N);
        }
        else{
            cblas_sgemm(Order, TransA, TransB, M, N, K,
					1, weight, K, processed_input_feature_map, N,
					0, output_feature_map, N);
        }
		
		if(relu){
			for(int output_index = 0; output_index < output_length; output_index++){
				if(with_bias){
				output_feature_map[output_index]+=
					bias[output_index / (height_output * width_output)];
				}
				output_feature_map[output_index] = output_feature_map[output_index] >= 0?
										output_feature_map[output_index] : 0;
			}
		}
		else{
			if(with_bias){
			for(int output_index = 0; output_index < output_length; output_index++){
				output_feature_map[output_index] += bias[output_index / (height_output * width_output)];
			}
			}
		}
		delete[] processed_input_feature_map;
		delete input_feature_map;
		height_input = height_output;
		width_input = width_output;
		return output_feature_map;
	}

private:
  int num_input;
  int num_output;
  int pad_h;
  int pad_w;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  float* weight;
  float* bias;
  bool with_bias;
  bool is_depthwise;
  int is_relu;

};

class ip{
public:
	

	//构造函数
	ip(int num_input_, int num_output_, float* weight_, float* bias_, bool with_bias_ = true){
		//initialization begin
		num_input = num_input_;
		num_output = num_output_;

		weight = weight_;
		bias = bias_;

	}
	ip(){}

	//层初始化
	void set(int num_input_, int num_output_,
				float* weight_, float* bias_, bool with_bias_ = true){
		//initialization begin
		num_input = num_input_;
		num_output = num_output_;

		weight = weight_;
        with_bias = with_bias_;
		bias = bias_;

	}

	//前向计算
	float* ip_forward(float* input_feature_map, int width_input, bool from_lstm = false){
		
		float* output_feature_map = new float[width_input * num_output];
		//M: A's row, C's row
		//N: B's col, C's col
		//K: A's col, B's row
		if(from_lstm){
			for(int i=0; i<width_input; i++){
				memcpy(output_feature_map + i * num_output, bias, num_output*sizeof(float));
			}
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
						width_input, num_output, num_input,
						1, input_feature_map, width_input, weight, num_input,
						1, output_feature_map, num_output);
			}
			else{
                if(with_bias){
					for(int i=0; i<width_input; i++){
						memcpy(output_feature_map + i * num_output, bias, num_output*sizeof(float));
					}
			        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
						width_input, num_output, num_input,
						1, input_feature_map, num_input, weight, num_input,
						1, output_feature_map, num_output);
                }else{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						width_input, num_output, num_input,
						1, input_feature_map, num_input, weight, num_output,
						0, output_feature_map, num_output);
                }
			
		}
//		delete input_feature_map;
		return output_feature_map;
	}

private:
	int num_input;
    int num_output;

    float* weight;
    float* bias;
    bool with_bias;
	int is_compressed;
};

class leakyrelu{

public:
	leakyrelu(int num_output_, float negative_slope_){
		num_output = num_output_;
		negative_slope = negative_slope_;
	}
	leakyrelu(){}

	void set(int num_output_, float negative_slope_){
		num_output = num_output_;
		negative_slope = negative_slope_;
	}

	float* relu_forward(float* input_feature_map, int height_output, int width_output){
		int length = height_output * width_output * num_output;
		float* output_feature_map = new float[length];
		for(int i = 0; i < length; i++){
			output_feature_map[i] = input_feature_map[i] < 0 ? (input_feature_map[i] * negative_slope) : input_feature_map[i];
		}
		delete input_feature_map;
		return output_feature_map;
	}

private:
	//picture properties
	int num_output;
	float negative_slope;

};


class pooling{
public:

	//构造函数
	pooling(int num_input_, int num_output_,
				int pad_h_, int pad_w_, int kernel_h_, int kernel_w_,
				int stride_h_, int stride_w_){

		num_input = num_input_;
		num_output = num_output_;
		pad_h = pad_h_;
		pad_w = pad_w_;
		kernel_h = kernel_h_;
		kernel_w = kernel_w_;
		stride_h = stride_h_;
		stride_w = stride_w_;
	}

	pooling(){}

	//层初始化
	void set(int num_input_, int num_output_,
				int pad_h_, int pad_w_, int kernel_h_, int kernel_w_,
				int stride_h_, int stride_w_){

		num_input = num_input_;
		num_output = num_output_;
		pad_h = pad_h_;
		pad_w = pad_w_;
		kernel_h = kernel_h_;
		kernel_w = kernel_w_;
		stride_h = stride_h_;
		stride_w = stride_w_;
	}

	void set(FILE* &fp)
	{
		fread(&num_input, sizeof(int), 1, fp);
		fread(&num_output, sizeof(int), 1, fp);
		fread(&kernel_h, sizeof(int), 1, fp);
		fread(&kernel_w, sizeof(int), 1, fp);
		fread(&pad_h, sizeof(int), 1, fp);
		fread(&pad_w, sizeof(int), 1, fp);
		fread(&stride_h, sizeof(int), 1, fp);
		fread(&stride_w, sizeof(int), 1, fp);
	}

	//前向计算
	float* pooling_forward(float* input_feature_map, int& height_input, int& width_input){

		int height_output = floor(float(height_input + pad_h * 2 - kernel_h) / stride_h +1);
		int width_output = floor(float(width_input + pad_w * 2 - kernel_w) / stride_w + 1);
		// cout<<"height_output compute:"<<float(height_input + pad_h * 2 - kernel_h) / stride_h<<endl;
		int length = height_output * width_output * num_output;
		float* output_feature_map = new float[length];
		//important initialization
		for(int i = 0; i < length; i++){
		output_feature_map[i] = -10000;   //attention: everyday's init is different
		}

		int pool_index, original_index;
		int hstart, wstart, hend, wend;
		for(int channels = 0; channels < num_output; channels++){
		for(int ph = 0; ph < height_output; ph++){
			for(int pw = 0; pw < width_output; pw++){
			pool_index = height_output * width_output * channels + ph * width_output + pw;
			hstart = ph * stride_h - pad_h;
			wstart = pw * stride_w - pad_w;
			hend = min(hstart + kernel_h, height_input);
			wend = min(wstart + kernel_w, width_input);
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			for(int h = hstart; h < hend; h++){
				for(int w = wstart; w < wend; w++){
				original_index = height_input * width_input * channels + h * width_input + w;
				if(input_feature_map[original_index] > output_feature_map[pool_index]){
					output_feature_map[pool_index] = input_feature_map[original_index];
				}
				}
			}
			}
		}
		}
		height_input = height_output;
		width_input = width_output;
		delete input_feature_map;
		return output_feature_map;
	}

private:
	//picture properties
	int num_input;
	int num_output;
	int height_input;
	int height_output;
	int width_input;
	int width_output;
	//for operation
	int pad_h;
	int pad_w;
	int kernel_h;
	int kernel_w;
	int stride_h;
	int stride_w;

};


class relu{

public:
	
	relu(int num_output_){
		num_output = num_output_;
	}
	relu(){}
	void set(int num_output_){
		num_output = num_output_;
	}

	void set(FILE* &fp)
	{
		fread(&num_output, sizeof(int), 1, fp);
	}

	float* relu_forward(float* input_feature_map, int height_output, int width_output){
	
		int length = height_output * width_output * num_output;
		float* output_feature_map = new float[length];
		for(int i = 0; i < length; i++){
			output_feature_map[i] = input_feature_map[i] < 0 ? 0 : input_feature_map[i];
		}
		delete input_feature_map;
		return output_feature_map;
	}

private:
	int num_output;
};


class softmax{
public:
	softmax(const int class_num_ )
	{
		class_num = class_num_;
	}
    softmax(){
    }
    void set(const int class_num_ )
    {
        class_num = class_num_;
    }
	float * softmax_forward(float* input_feature_map, int height_output, int width_output)
	{
		int total_num = height_output * width_output * class_num;
    	float* output_feature_map = new float[total_num];
    	int channel_num = height_output * width_output;
		for(int it = 0 ; it < channel_num ; it++)
    	{	float cur_max = -1e10;
    		for(int ite = 0 ; ite < class_num ; ite++)
    		{
				if (cur_max < input_feature_map[ite + class_num * it])
					cur_max = input_feature_map[ite + class_num * it];
    		}
     		for(int ite = 0 ; ite < class_num ; ite++)
    		{
				input_feature_map[ite + class_num * it] = input_feature_map[ite + class_num * it] - cur_max;
			}
    	}
    	float channel_sum;
    	for(int it = 0 ; it < channel_num ; it++)
    	{	channel_sum = 0 ;
    		for(int ite = 0 ; ite < class_num ; ite++)
    		{
    			channel_sum += exp(input_feature_map[ite + class_num * it]);
    		}
     		for(int ite = 0 ; ite < class_num ; ite++)
    		{
    			output_feature_map[ite + class_num * it] = exp(input_feature_map[ite + class_num * it]) / channel_sum;
    		}
    	}
    	delete[] input_feature_map;
    	return output_feature_map;
	}
private:
	int class_num;
};


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


class OLHCT
{
public:

    OLHCT(){};

    int load_param(int pre_d_model_, int d_model_1_, int d_model_2_ , int n_lm_layer_, int fc_svd_dims_, int em_svd_dims_, int transformer_svd_dims_,int classes_);
    int load_model(AAsset *fp);
    int release();
    void set();
    vector< Results > forward(float* src, int bs);
    int transpose(float* src, int height, int width);
    int matrix_split(float* src, float* dest, int height, int width, int start, int len);
    int matrix_split_transpose(float* src, float* dest, int height, int width, int start, int len);
    int softmax(float* src, int height, int width);
    int layer_norm(float* src, int bs, int in_c, float* gamma, float* beta);


public:
    int pre_d_model;
    int d_model_1;
    int d_model_2;
    int n_lm_layer;
    int fc_svd_dims;
    int em_svd_dims;
    int transformer_svd_dims;
    int classes;


    float* pre_conv_layers_0_0_weight;
    float* pre_conv_layers_0_0_bias;

    float* pre_conv_layers_1_0_weight;
    float* pre_conv_layers_1_0_bias;

    float* encoder_layers_0_self_attn_weight;
    float* encoder_layers_0_self_attn_q_weight;
    float* encoder_layers_0_self_attn_k_weight;
    float* encoder_layers_0_self_attn_v_weight;
    float* encoder_layers_0_self_attn_q_bias;
    float* encoder_layers_0_self_attn_k_bias;
    float* encoder_layers_0_self_attn_v_bias;
    float* encoder_layers_0_self_attn_out_weight;
    float* encoder_layers_0_self_attn_out_bias;
    float* encoder_layers_0_self_attn_linear1_weight;
    float* encoder_layers_0_self_attn_linear1_bias;
    float* encoder_layers_0_self_attn_linear2_weight;
    float* encoder_layers_0_self_attn_linear2_bias;
    float* encoder_layers_0_self_attn_norm1_weight;
    float* encoder_layers_0_self_attn_norm1_bias;
    float* encoder_layers_0_self_attn_norm2_weight;
    float* encoder_layers_0_self_attn_norm2_bias;

    float* encoder_layers_1_self_attn_weight;
    float* encoder_layers_1_self_attn_q_weight;
    float* encoder_layers_1_self_attn_k_weight;
    float* encoder_layers_1_self_attn_v_weight;
    float* encoder_layers_1_self_attn_q_bias;
    float* encoder_layers_1_self_attn_k_bias;
    float* encoder_layers_1_self_attn_v_bias;
    float* encoder_layers_1_self_attn_out_weight;
    float* encoder_layers_1_self_attn_out_bias;
    float* encoder_layers_1_self_attn_linear1_svd0_weight;
    float* encoder_layers_1_self_attn_linear1_svd1_weight;
    float* encoder_layers_1_self_attn_linear1_svd1_bias;
    float* encoder_layers_1_self_attn_linear2_svd0_weight;
    float* encoder_layers_1_self_attn_linear2_svd1_weight;
    float* encoder_layers_1_self_attn_linear2_svd1_bias;
    float* encoder_layers_1_self_attn_norm1_weight;
    float* encoder_layers_1_self_attn_norm1_bias;
    float* encoder_layers_1_self_attn_norm2_weight;
    float* encoder_layers_1_self_attn_norm2_bias;

    float* norm_layers_0_weight;
    float* norm_layers_0_bias;
    float* norm_layers_1_weight;
    float* norm_layers_1_bias;

    float* post_conv_layers_0_0_weight;
    float* post_conv_layers_0_0_bias;

    float* post_conv_layers_1_0_weight;
    float* post_conv_layers_1_0_bias;


    float* classifier_svd0_weight;
    float* classifier_svd1_weight;
    float* classifier_svd1_bias;

    float* lm_encoder_layers_0_self_attn_weight;
    float* lm_encoder_layers_0_self_attn_q_weight;
    float* lm_encoder_layers_0_self_attn_k_weight;
    float* lm_encoder_layers_0_self_attn_v_weight;
    float* lm_encoder_layers_0_self_attn_q_bias;
    float* lm_encoder_layers_0_self_attn_k_bias;
    float* lm_encoder_layers_0_self_attn_v_bias;
    float* lm_encoder_layers_0_self_attn_out_weight;
    float* lm_encoder_layers_0_self_attn_out_bias;
    float* lm_encoder_layers_0_self_attn_linear1_svd0_weight;
    float* lm_encoder_layers_0_self_attn_linear1_svd1_weight;
    float* lm_encoder_layers_0_self_attn_linear1_svd1_bias;
    float* lm_encoder_layers_0_self_attn_linear2_svd0_weight;
    float* lm_encoder_layers_0_self_attn_linear2_svd1_weight;
    float* lm_encoder_layers_0_self_attn_linear2_svd1_bias;
    float* lm_encoder_layers_0_self_attn_norm1_weight;
    float* lm_encoder_layers_0_self_attn_norm1_bias;
    float* lm_encoder_layers_0_self_attn_norm2_weight;
    float* lm_encoder_layers_0_self_attn_norm2_bias;

    float* lm_encoder_layers_1_self_attn_weight;
    float* lm_encoder_layers_1_self_attn_q_weight;
    float* lm_encoder_layers_1_self_attn_k_weight;
    float* lm_encoder_layers_1_self_attn_v_weight;
    float* lm_encoder_layers_1_self_attn_q_bias;
    float* lm_encoder_layers_1_self_attn_k_bias;
    float* lm_encoder_layers_1_self_attn_v_bias;
    float* lm_encoder_layers_1_self_attn_out_weight;
    float* lm_encoder_layers_1_self_attn_out_bias;
    float* lm_encoder_layers_1_self_attn_linear1_svd0_weight;
    float* lm_encoder_layers_1_self_attn_linear1_svd1_weight;
    float* lm_encoder_layers_1_self_attn_linear1_svd1_bias;
    float* lm_encoder_layers_1_self_attn_linear2_svd0_weight;
    float* lm_encoder_layers_1_self_attn_linear2_svd1_weight;
    float* lm_encoder_layers_1_self_attn_linear2_svd1_bias;
    float* lm_encoder_layers_1_self_attn_norm1_weight;
    float* lm_encoder_layers_1_self_attn_norm1_bias;
    float* lm_encoder_layers_1_self_attn_norm2_weight;
    float* lm_encoder_layers_1_self_attn_norm2_bias;

    float* lm_encoder_layers_2_self_attn_weight;
    float* lm_encoder_layers_2_self_attn_q_weight;
    float* lm_encoder_layers_2_self_attn_k_weight;
    float* lm_encoder_layers_2_self_attn_v_weight;
    float* lm_encoder_layers_2_self_attn_q_bias;
    float* lm_encoder_layers_2_self_attn_k_bias;
    float* lm_encoder_layers_2_self_attn_v_bias;
    float* lm_encoder_layers_2_self_attn_out_weight;
    float* lm_encoder_layers_2_self_attn_out_bias;
    float* lm_encoder_layers_2_self_attn_linear1_svd0_weight;
    float* lm_encoder_layers_2_self_attn_linear1_svd1_weight;
    float* lm_encoder_layers_2_self_attn_linear1_svd1_bias;
    float* lm_encoder_layers_2_self_attn_linear2_svd0_weight;
    float* lm_encoder_layers_2_self_attn_linear2_svd1_weight;
    float* lm_encoder_layers_2_self_attn_linear2_svd1_bias;
    float* lm_encoder_layers_2_self_attn_norm1_weight;
    float* lm_encoder_layers_2_self_attn_norm1_bias;
    float* lm_encoder_layers_2_self_attn_norm2_weight;
    float* lm_encoder_layers_2_self_attn_norm2_bias;

    float* lm_encoder_layers_3_self_attn_weight;
    float* lm_encoder_layers_3_self_attn_q_weight;
    float* lm_encoder_layers_3_self_attn_k_weight;
    float* lm_encoder_layers_3_self_attn_v_weight;
    float* lm_encoder_layers_3_self_attn_q_bias;
    float* lm_encoder_layers_3_self_attn_k_bias;
    float* lm_encoder_layers_3_self_attn_v_bias;
    float* lm_encoder_layers_3_self_attn_out_weight;
    float* lm_encoder_layers_3_self_attn_out_bias;
    float* lm_encoder_layers_3_self_attn_linear1_svd0_weight;
    float* lm_encoder_layers_3_self_attn_linear1_svd1_weight;
    float* lm_encoder_layers_3_self_attn_linear1_svd1_bias;
    float* lm_encoder_layers_3_self_attn_linear2_svd0_weight;
    float* lm_encoder_layers_3_self_attn_linear2_svd1_weight;
    float* lm_encoder_layers_3_self_attn_linear2_svd1_bias;
    float* lm_encoder_layers_3_self_attn_norm1_weight;
    float* lm_encoder_layers_3_self_attn_norm1_bias;
    float* lm_encoder_layers_3_self_attn_norm2_weight;
    float* lm_encoder_layers_3_self_attn_norm2_bias;

    float* lm_embedding_svd0_weight;
    float* lm_embedding_svd1_weight;

    conv1d pre_conv_0_0;
    Transformer encoder_0;
    conv1d post_conv_0_0;

    conv1d pre_conv_1_0;
    Transformer_SVD encoder_1;
    conv1d post_conv_1_0;

    ip lm_embedding_svd0;
    ip lm_embedding_svd1;

    Transformer_SVD lm_encoder_0;
    Transformer_SVD lm_encoder_1;
    Transformer_SVD lm_encoder_2;
    Transformer_SVD lm_encoder_3;

    ip classifier_svd0;
    ip classifier_svd1;



};


static vector<vector<pair<int,float>>> CRUD(float* lm_input, vector<float>& confidence_array, int points_num, int classes);


#endif //X86CONV_TRANSFORMER_H
