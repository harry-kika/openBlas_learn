//
// Created by Levi Lin on 3/5/22.
//


#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

vector<vector<pair<int,float>>> CRUD(float* lm_input, vector<float>& confidence_array, int points_num, int classes){
    vector<float> tmp_vec;
    vector<pair<int,float>> net_token_top1;
    vector<pair<int,float>> token_top1;
    vector<pair<int,float>> token_top2;
    vector<pair<int,float>> token_blank;

    float* new_lm_vec = new float[(classes+1)]();


    for (int i = 0; i < points_num; i++)
    {
        memcpy(new_lm_vec, lm_input + i*(classes+1), (classes+1) * sizeof(float));
        vector<float> lm_vec(new_lm_vec, new_lm_vec+(classes+1));

        int lm_maxPosition = max_element(lm_vec.begin(), lm_vec.end()) - lm_vec.begin();
        pair<int,float> top1(lm_maxPosition,lm_vec[lm_maxPosition]);
        pair<int,float> blank(0,lm_vec[0]);

        int near_num;
        if(lm_maxPosition>classes-4){
            near_num = -2;
        }
        else{
            near_num = 2;
        }

        pair<int,float> top1_near(lm_maxPosition+near_num,lm_vec[lm_maxPosition+near_num]);
        pair<int,float> top1_near2(lm_maxPosition+near_num-1,lm_vec[lm_maxPosition+near_num-1]);

        if(lm_vec[lm_maxPosition]>0.9){
            token_top2.push_back(top1_near);
        }
        else if(lm_vec[lm_maxPosition]>0.8){
            token_top2.push_back(top1_near2);
        }
        else{
            tmp_vec = lm_vec;
            sort(tmp_vec.begin(), tmp_vec.end());
            int lm_top2Position = find(lm_vec.begin(), lm_vec.end(), tmp_vec[classes-1]) - lm_vec.begin();
            pair<int,float> top2(lm_top2Position,tmp_vec[classes-1]);

            token_top2.push_back(top2);
        }


        token_blank.push_back(blank);
        token_top1.push_back(top1);
        vector<float>().swap(tmp_vec);
        vector<float>().swap(lm_vec);
    }

    vector<pair<int,float>> tmp_result;
    vector<vector<pair<int,float>>> multiple_results;
    float confidence = 0.0;
    int count =0;


    //lm greedy search
    for(int i=0;i<token_top1.size();i++){
        confidence = confidence + log(token_top1[i].second);
    }
    confidence = exp(confidence);
    multiple_results.push_back(token_top1);
    confidence_array.push_back(confidence);
    confidence = 0.0;


    //新增字符
    vector<int> add_num = {1};
    if(points_num>30){
        add_num.push_back(2);
    }
    if(points_num>48){
        add_num.push_back(3);
    }

    for(int k=0;k<add_num.size();k++){
        tmp_result = token_top1;
        for(int i=0;i<token_top1.size();i++){
            if(token_top1[i].first == 0){
                if(count < add_num[k]){
                    if(token_top1[i].second <1){
                        tmp_result[i] = token_top2[i];
                        count = count+1;
                    }
                }
                else{
                    break;
                }
            }
        }
        for(int i=0;i<tmp_result.size();i++){
            confidence = confidence + log(tmp_result[i].second);
        }
        confidence = exp(confidence);

        multiple_results.push_back(tmp_result);
        confidence_array.push_back(confidence);
        tmp_result.clear();
        confidence = 0.0;
        count = 0;
    }



    //修改字符(0)
    vector<int> replace_num = {1};
    if(points_num>20){
        replace_num.push_back(2);
    }
    if(points_num>35){
        replace_num.push_back(3);
    }
    if(points_num>45){
        replace_num.push_back(4);
    }

    for(int k=0;k<replace_num.size();k++){
        tmp_result = token_top1;
        for(int i=0;i<token_top1.size();i++){
            if(token_top1[i].first != 0){
                if(count < replace_num[k]){
                    if(token_top1[i].second <1){
                        tmp_result[i] = token_top2[i];
                        count = count+1;
                    }
                }
                else{
                    break;
                }
            }
        }
        for(int i=0;i<tmp_result.size();i++){
            confidence = confidence + log(tmp_result[i].second);
        }
        confidence = exp(confidence);

        multiple_results.push_back(tmp_result);
        confidence_array.push_back(confidence);
        tmp_result.clear();
        confidence = 0.0;
        count = 0;
    }


    //修改字符(1)
    for(int k=0;k<replace_num.size();k++){
        tmp_result = token_top1;
        for(int i=0;i<token_top1.size();i++){
            if(token_top1[token_top1.size()-i-1].first != 0){
                if(count < replace_num[k]){
                    if(token_top1[token_top1.size()-i-1].second <1){
                        tmp_result[token_top1.size()-i-1] = token_top2[token_top1.size()-i-1];
                        count = count+1;
                    }
                }
                else{
                    break;
                }
            }
        }
        for(int i=0;i<tmp_result.size();i++){
            confidence = confidence + log(tmp_result[i].second);
        }
        confidence = exp(confidence);

        multiple_results.push_back(tmp_result);
        confidence_array.push_back(confidence);
        tmp_result.clear();
        confidence = 0.0;
        count = 0;
    }


    //删除字符(0)
    vector<int> del_num = {1};
    if(points_num>35){
        del_num.push_back(2);
    }
    if(points_num>50){
        del_num.push_back(3);
    }
    for(int k=0;k<del_num.size();k++){
        tmp_result = token_top1;
        for(int i=0;i<token_top1.size();i++){
            if(token_top1[i].first != 0){
                if(count < del_num[k]){
                    if(token_top1[i].second <0.999){
                        tmp_result[i].first = 0;
                        tmp_result[i].second = token_blank[i].second;
                        count = count+1;
                    }
                }
                else{
                    break;
                }
            }
        }
        for(int i=0;i<tmp_result.size();i++){
            confidence = confidence + log(tmp_result[i].second);
        }
        confidence = exp(confidence);

        multiple_results.push_back(tmp_result);
        confidence_array.push_back(confidence);
        tmp_result.clear();
        confidence = 0.0;
        count = 0;
    }

    //删除字符(1)
    for(int k=0;k<del_num.size();k++){
        tmp_result = token_top1;
        for(int i=0;i<token_top1.size();i++){
            if(token_top1[token_top1.size()-i-1].first != 0){
                if(count < del_num[k]){
                    if(token_top1[token_top1.size()-i-1].second <0.999){
                        tmp_result[token_top1.size()-i-1].first = 0;
                        tmp_result[token_top1.size()-i-1].second = token_blank[token_top1.size()-i-1].second;
                        count = count+1;
                    }
                }
                else{
                    break;
                }
            }
        }
        for(int i=0;i<tmp_result.size();i++){
            confidence = confidence + log(tmp_result[i].second);
        }
        confidence = exp(confidence);

        multiple_results.push_back(tmp_result);
        confidence_array.push_back(confidence);
        tmp_result.clear();
        confidence = 0.0;
        count = 0;
    }

    float sum_confidence = 0;
    for(int i=0;i<confidence_array.size();i++){
        sum_confidence += confidence_array[i];
    }
    for(int i=0;i<confidence_array.size();i++){
        confidence_array[i] = confidence_array[i] / sum_confidence;
    }

    delete[] new_lm_vec;
    vector<float>().swap(tmp_vec);
    vector<pair<int,float>>().swap(net_token_top1);
    vector<pair<int,float>>().swap(token_top1);
    vector<pair<int,float>>().swap(token_top2);
    vector<pair<int,float>>().swap(token_blank);
    vector<pair<int,float>>().swap(tmp_result);
    vector<int>().swap(add_num);
    vector<int>().swap(replace_num);
    vector<int>().swap(del_num);


    return multiple_results;
}