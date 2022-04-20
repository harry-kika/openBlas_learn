//
// Created by Harry on 2022/4/14.
//

#ifndef OPENBLAS_LEARN_NATIVE_H
#define OPENBLAS_LEARN_NATIVE_H

#include "Utils.h"
#include <time.h>
#include <map>
#include <string>
#include "openblas/cblas.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <ctime>
#include "source/olhct.h"

#define ASSERT(status, ret)     if (!(status)) { return ret; }

using namespace std;

OLHCT Net;
static vector<float> outputs_confidence;
static vector<string> lm_result_array;
static vector<string> dict_27769;


static vector<string> split_string(const string& str)
{
    vector<string> strings;
    string delimiter = "";

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;

    int idx=str.find("\r\n");
    if(idx == string::npos)
        delimiter = "\n";
    else
        delimiter = "\r\n";

    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

void getRotationMatrix2D_our(float center_x,float center_y ,double angle, double scale, float* Mat_M)
{
    angle *= M_PI/180;
    double alpha = cos(angle)*scale;
    double beta = sin(angle)*scale;

//    float* Mat_M = new float[3 * 3]();

    Mat_M[0] = alpha;
    Mat_M[1] = beta;
    Mat_M[2] = (1-alpha)*center_x - beta*center_y;
    Mat_M[3] = -beta;
    Mat_M[4] = alpha;
    Mat_M[5] = beta*center_x + (1-alpha)*center_y;

    /////// external
    Mat_M[6] = 0.0;
    Mat_M[7] = 0.0;
    Mat_M[8] = 1.0;

}

void transpose(float *src, int height, int width) {
    // convert A : a*b -> b*a
    float* tmp = new float[width * height];
    for(int i=0; i<width; i++){
        for(int j=0; j<height; j++){
            tmp[i * height + j] = src[j * width + i];
        }
    }
    memcpy(src, tmp, height * width *sizeof(float));
    delete[] tmp;

}

float get_array_max(float *src, int size){
    int count;
    int highest;
    highest = src[0];
    for (count = 1; count < size; count++)
    {
        if (src[count] > highest)
            highest = src[count];
    }

    return highest;
}

float get_array_min(float *src, int size){
    int count;
    int lowest;
    lowest = src[0];
    for (count = 1; count < size; count++)
    {
        if (src[count] < lowest){
            lowest = src[count];
        }
    }

    return lowest;
}

float LinearRegression(float* x, float* y, int points_num) {

    // Y = X*W
    // W = (Xt * X)-1 * Xt * Y
    float* x_points_pad = new float[points_num * 2]();
    float* x_points_pad_transpose = new float[points_num * 2]();
    for (int i = 0; i < points_num * 2; i++) {
        if (i % 2 == 0) {
            x_points_pad[i] = x[i / 2];
        }
        else {
            x_points_pad[i] = 1.0f;
        }
    }
    for (int i = 0; i < points_num * 2; i++) {
        if (i < points_num) {
            x_points_pad_transpose[i] = x[i];
        }
        else {
            x_points_pad_transpose[i] = 1.0f;
        }
    }

    //Xt * X
    int M = 2;
    int N = 2;
    int K = points_num;
    float* first_output = new float[4]();
    float* second_output = new float[points_num * 2]();
    float* theta_output = new float[2]();

    CBLAS_ORDER Order = CblasRowMajor;
    CBLAS_TRANSPOSE TransA = CblasNoTrans;
    CBLAS_TRANSPOSE TransB = CblasNoTrans;

    cblas_sgemm(Order, TransA, TransB, M, N, K,
                1, x_points_pad_transpose, K, x_points_pad, N,
                0, first_output, N);

    //(Xt * X)-1  二阶矩阵求逆

    double tmp_a = (double)first_output[0];
    double tmp_b = (double)first_output[1];
    double tmp_c = (double)first_output[2];
    double tmp_d = (double)first_output[3];

    float Determinant = tmp_a * tmp_d - tmp_b * tmp_c;
    if(Determinant == 0){
        Determinant = 1;
    }

    first_output[0] = tmp_d / Determinant;
    first_output[3] = tmp_a / Determinant;
    first_output[1] = -first_output[1] / Determinant;
    first_output[2] = -first_output[2] / Determinant;

    //(Xt * X)-1 * Xt
    M = 2;
    N = points_num;
    K = 2;
    cblas_sgemm(Order, TransA, TransB, M, N, K,
                1, first_output, K, x_points_pad_transpose, N,
                0, second_output, N);

    //W = (Xt * X)-1 * Xt * Y
    M = 2;
    N = 1;
    K = points_num;
    cblas_sgemm(Order, TransA, TransB, M, N, K,
                1, second_output, K, y, N,
                0, theta_output, N);


    delete[] x_points_pad;
    delete[] x_points_pad_transpose;
    delete[] first_output;
    delete[] second_output;

    return theta_output[0];
}

float* RotateToHorizontal(float* get_input_points, int pointsNum) {

    float* x_points = new float[pointsNum]();
    float* y_points = new float[pointsNum]();
    float* s_points = new float[pointsNum]();

    for (int i = 0; i < pointsNum; i++) {
        x_points[i] = get_input_points[i * 3 + 0];
        y_points[i] = get_input_points[i * 3 + 1];
        s_points[i] = get_input_points[i * 3 + 2];
    }


    float theta = LinearRegression(x_points, y_points, pointsNum);
    float Rad_to_deg = 45.0 / atan(1.0);
    float angle = atan(theta) * Rad_to_deg;

    float xmin = get_array_min(x_points, pointsNum);
    float xmax = get_array_max(x_points, pointsNum);
    float points_width = xmax - xmin + 1;
    float ymin = get_array_min(y_points, pointsNum);
    float ymax = get_array_max(y_points, pointsNum);
    float points_height = ymax - ymin + 1;

    float* M_tmp = new float[3 * 3]();
    getRotationMatrix2D_our(points_width/2, points_height/2, angle, 1, M_tmp);

    float* points_coor = new float[pointsNum * 3]();
    for (int i = 0; i < pointsNum; i++) {
        points_coor[i * 3 + 0] = get_input_points[i * 3 + 0];
        points_coor[i * 3 + 1] = get_input_points[i * 3 + 1];
        points_coor[i * 3 + 2] = 1.0;
    }

    int M = 3;
    int N = pointsNum;
    int K = 3;
    float* points_coor_rotated = new float[N * M]();

    CBLAS_ORDER Order = CblasRowMajor;
    CBLAS_TRANSPOSE TransA = CblasNoTrans;
    CBLAS_TRANSPOSE TransB = CblasTrans;

    cblas_sgemm(Order, TransA, TransB, M, N, K,
                1, M_tmp, K, points_coor, K,
                0, points_coor_rotated, N);

    transpose(points_coor_rotated, 3, pointsNum);

    float* R2H_points = new float[pointsNum * 3]();
    for (int i = 0; i < pointsNum; i++) {
        R2H_points[i * 3 + 0] = points_coor_rotated[i * 3 + 0];
        R2H_points[i * 3 + 1] = points_coor_rotated[i * 3 + 1];
        R2H_points[i * 3 + 2] = get_input_points[i * 3 + 2];
    }

    delete[] x_points;
    delete[] y_points;
    delete[] s_points;
    delete[] M_tmp;
    delete[] points_coor;
    delete[] points_coor_rotated;

    return R2H_points;
}

float* NormalizePointsAdvanced(float* points, int pointsNum, bool is_forLine) {

    float len_, px_L, py_L, dx_L, dy_L;
    vector<float> Px_L; vector<float> Py_L; vector<float> Lens; vector<float> Dx_L;
    vector<float> Dy_L;

    if (pointsNum == 1) {
        float u_x = 0.0f;
        float u_y = 0.0f;
        float delta_x = 1.0f;
        float delta_y = 1.0f;

        float* NP_points = new float[pointsNum * 3]();

        for (int i = 0; i < pointsNum; i++) {
            if (is_forLine) {
                NP_points[i * 3 + 0] = (points[i * 3 + 0] - u_x) / delta_y;
                NP_points[i * 3 + 1] = (points[i * 3 + 1] - u_y) / delta_y;
                NP_points[i * 3 + 2] = points[i * 3 + 2];
            }
            else {
                NP_points[i * 3 + 0] = (points[i * 3 + 0] - u_x) / delta_x;
                NP_points[i * 3 + 1] = (points[i * 3 + 1] - u_y) / delta_x;
                NP_points[i * 3 + 2] = points[i * 3 + 2];
            }
        }

        vector<float>().swap(Px_L);
        vector<float>().swap(Py_L);
        vector<float>().swap(Lens);
        vector<float>().swap(Dx_L);
        vector<float>().swap(Dy_L);

        return NP_points;
    }
    else {
        float* x_points = new float[pointsNum - 1]();
        float* y_points = new float[pointsNum - 1]();
        float* s_points = new float[pointsNum - 1]();
        float* xa1_points = new float[pointsNum - 1]();
        float* ya1_points = new float[pointsNum - 1]();
        float* sa1_points = new float[pointsNum - 1]();

        for (int i = 0; i < pointsNum - 1; i++) {
            x_points[i] = points[i * 3 + 0];
            y_points[i] = points[i * 3 + 1];
            s_points[i] = points[i * 3 + 2];
            xa1_points[i] = points[(i + 1) * 3 + 0];
            ya1_points[i] = points[(i + 1) * 3 + 1];
            sa1_points[i] = points[(i + 1) * 3 + 2];
            if (s_points[i] == sa1_points[i]) {
                len_ = sqrt(pow(xa1_points[i] - x_points[i], 2) + pow(ya1_points[i] - y_points[i], 2));
                px_L = len_ * (x_points[i] + xa1_points[i]) * 0.5;
                py_L = len_ * (y_points[i] + ya1_points[i]) * 0.5;
                Px_L.push_back(px_L);
                Py_L.push_back(py_L);
                Lens.push_back(len_);
            }
        }
        float sum_Px_L = 0;
        float sum_Lens = 0;
        float sum_Py_L = 0;
        float sum_Dx_L = 0;
        float sum_Dy_L = 0;
        for (int i = 0; i < Px_L.size(); i++) {
            sum_Px_L = sum_Px_L + Px_L[i];
        }
        for (int i = 0; i < Py_L.size(); i++) {
            sum_Py_L = sum_Py_L + Py_L[i];
        }
        for (int i = 0; i < Lens.size(); i++) {
            sum_Lens = sum_Lens + Lens[i];
        }
        float u_x = sum_Px_L / sum_Lens;
        float u_y = sum_Py_L / sum_Lens;


        for (int i = 0; i < pointsNum - 1; i++) {
            if (s_points[i] == sa1_points[i]) {
                len_ = sqrt(pow(x_points[i] - xa1_points[i], 2) + pow(y_points[i] - ya1_points[i], 2));
                dx_L = len_ / 3 * (pow(xa1_points[i] - u_x, 2) + pow(x_points[i] - u_x, 2) + (xa1_points[i] - u_x) * (x_points[i] - u_x));
                dy_L = len_ / 3 * (pow(ya1_points[i] - u_y, 2) + pow(y_points[i] - u_y, 2) + (ya1_points[i] - u_y) * (y_points[i] - u_y));
                Dx_L.push_back(dx_L);
                Dy_L.push_back(dy_L);
            }
        }
        for (int i = 0; i < Dx_L.size(); i++) {
            sum_Dx_L = sum_Dx_L + Dx_L[i];
        }
        for (int i = 0; i < Dy_L.size(); i++) {
            sum_Dy_L = sum_Dy_L + Dy_L[i];
        }
        float delta_x = sqrt(sum_Dx_L / sum_Lens);
        float delta_y = sqrt(sum_Dy_L / sum_Lens);
        if (delta_x < 0.009) {
            delta_x = 1.0;
        }
        if (delta_y < 0.009) {
            delta_y = 1.0;
        }

        float* NP_points = new float[pointsNum * 3]();


        for (int i = 0; i < pointsNum; i++) {
            if (is_forLine) {
                NP_points[i * 3 + 0] = (points[i * 3 + 0] - u_x) / delta_y;
                NP_points[i * 3 + 1] = (points[i * 3 + 1] - u_y) / delta_y;
                NP_points[i * 3 + 2] = points[i * 3 + 2];
            }
            else {
                NP_points[i * 3 + 0] = (points[i * 3 + 0] - u_x) / delta_x;
                NP_points[i * 3 + 1] = (points[i * 3 + 1] - u_y) / delta_x;
                NP_points[i * 3 + 2] = points[i * 3 + 2];
            }
        }

        delete[] x_points;
        delete[] y_points;
        delete[] s_points;
        delete[] xa1_points;
        delete[] ya1_points;
        delete[] sa1_points;

        vector<float>().swap(Px_L);
        vector<float>().swap(Py_L);
        vector<float>().swap(Lens);
        vector<float>().swap(Dx_L);
        vector<float>().swap(Dy_L);

        return NP_points;
    }
}

vector<float> SpeedNormalize(float* points, int& pointsNum, float density) {
    float* s_points = new float[pointsNum]();
    for (int i = 0; i < pointsNum; i++) {
        s_points[i] = points[i * 3 + 2];
    }
    float num_strokes = get_array_max(s_points, pointsNum) + 1;
    vector<float> strokes;
    vector<float> x_tmp_points;
    vector<float> y_tmp_points;
    float len = 0.0f;
    vector<float> lengths;
    vector<float> r;
    vector<float> r_interp;
    vector<float> p1_interp;
    vector<float> p2_interp;
    float linspace = 0.0f;


    for (int i = 0; i < int(num_strokes); i++) {
        for (int j = 0; j < pointsNum; j++) {
            if (s_points[j] == i) {
                x_tmp_points.push_back(points[j * 3 + 0]);
                y_tmp_points.push_back(points[j * 3 + 1]);
            }
        }

        for (int k = 1; k < x_tmp_points.size(); k++) {
            len = sqrt(pow(x_tmp_points[k] - x_tmp_points[k - 1], 2) + pow(y_tmp_points[k] - y_tmp_points[k - 1], 2));
            lengths.push_back(len);
        }

        if (x_tmp_points.size() == 1 && y_tmp_points.size() == 1) {
            r.push_back(0.0f);
            r_interp.push_back(0.0f);
        }
        else {
            r.push_back(0.0f);
            r.push_back(lengths[0]);

            for (int t = 1; t < lengths.size(); t++) {
                r.push_back(lengths[t] + r[t]);
            }

            linspace = r.back() / (round(r.back() / density) - 1);

            r_interp.push_back(0.0f);
            for (int n = 1; n < int(round(r.back() / density) - 1); n++) {
                r_interp.push_back(n * linspace);
            }
            r_interp.push_back(r.back());
        }


        for (int m = 0; m < r_interp.size(); m++) {
            float tmp_count = 0;
            float k1 = 0;
            float k2 = 0;
            float b1 = 0;
            float b2 = 0;
            if (x_tmp_points.size() == 1 && y_tmp_points.size() == 1) {
                p1_interp.push_back(x_tmp_points[0]);
                p2_interp.push_back(y_tmp_points[0]);
            }
            else {
                for (int n = 1; n < r.size(); n++) {
                    if (r_interp[m] < r[0]) {
                        p1_interp.push_back(x_tmp_points[0]);
                        p2_interp.push_back(y_tmp_points[0]);
                    }
                    else if (r_interp[m] >= r[n - 1] && r_interp[m] <= r[n]) {
                        tmp_count = n;
                        k1 = (x_tmp_points[tmp_count] - x_tmp_points[tmp_count - 1]) / (r[tmp_count] - r[tmp_count - 1]);
                        k2 = (y_tmp_points[tmp_count] - y_tmp_points[tmp_count - 1]) / (r[tmp_count] - r[tmp_count - 1]);
                        if (r[tmp_count] - r[tmp_count - 1] != 0) {
                            b1 = x_tmp_points[tmp_count] - k1 * r[tmp_count];
                            b2 = y_tmp_points[tmp_count] - k2 * r[tmp_count];
                            p1_interp.push_back(k1 * r_interp[m] + b1);
                            p2_interp.push_back(k2 * r_interp[m] + b2);
                        }
                    }
                    else if (r_interp[m] > r.back()) {
                        p1_interp.push_back(x_tmp_points.back());
                        p2_interp.push_back(y_tmp_points.back());
                    }
                }
            }

        }
        for (int t = 0; t < p1_interp.size(); t++) {
            strokes.push_back(p1_interp[t]);
            strokes.push_back(p2_interp[t]);
            strokes.push_back(i);
        }

        x_tmp_points.clear();
        y_tmp_points.clear();
        lengths.clear();
        r.clear();
        r_interp.clear();
        p1_interp.clear();
        p2_interp.clear();
    }

    pointsNum = int(strokes.size() / 3);

    delete[] s_points;
    vector<float>().swap(x_tmp_points);
    vector<float>().swap(y_tmp_points);
    vector<float>().swap(lengths);
    vector<float>().swap(r);
    vector<float>().swap(r_interp);
    vector<float>().swap(p1_interp);
    vector<float>().swap(p2_interp);

    return strokes;
}

float* LineFeature(vector<float> points, int& pointsNum) {
    float* x_points = new float[pointsNum - 1]();
    float* y_points = new float[pointsNum - 1]();
    float* s_points = new float[pointsNum - 1]();
    float* xa1_points = new float[pointsNum - 1]();
    float* ya1_points = new float[pointsNum - 1]();
    float* sa1_points = new float[pointsNum - 1]();
    float* x_shift = new float[pointsNum - 1]();
    float* y_shift = new float[pointsNum - 1]();
    vector<bool> pen_state_1;
    vector<bool> pen_state_2;

    float* output_points = new float[(pointsNum - 1 + 128) * 6]();

    for (int i = 0; i < pointsNum - 1; i++) {
        x_points[i] = points[i * 3 + 0];
        y_points[i] = points[i * 3 + 1];
        s_points[i] = points[i * 3 + 2];
        xa1_points[i] = points[(i + 1) * 3 + 0];
        ya1_points[i] = points[(i + 1) * 3 + 1];
        sa1_points[i] = points[(i + 1) * 3 + 2];
        x_shift[i] = xa1_points[i] - x_points[i];
        y_shift[i] = ya1_points[i] - y_points[i];
        if (sa1_points[i] == s_points[i]) {
            pen_state_1.push_back(1);
            pen_state_2.push_back(0);
        }
        else {
            pen_state_1.push_back(0);
            pen_state_2.push_back(1);
        }
    }

    for (int i = 0; i < pointsNum - 1; i++) {
        output_points[i * 6 + 0] = x_points[i];
        output_points[i * 6 + 1] = y_points[i];
        output_points[i * 6 + 2] = x_shift[i];
        output_points[i * 6 + 3] = y_shift[i];
        output_points[i * 6 + 4] = pen_state_1[i];
        output_points[i * 6 + 5] = pen_state_2[i];
    }
    pointsNum = pointsNum - 1 + 128;

    delete[] x_points;
    delete[] y_points;
    delete[] s_points;
    delete[] xa1_points;
    delete[] ya1_points;
    delete[] sa1_points;
    delete[] x_shift;
    delete[] y_shift;
    vector<bool>().swap(pen_state_1);
    vector<bool>().swap(pen_state_2);

    return output_points;
}

vector<string> PreRecognize(float* points, int& pointsNum) {
    bool x_equal = true;
    bool y_equal = true;
    vector<string> pre_outputs ={};

    if(pointsNum<=4){
        pre_outputs.push_back("。");
    }
    else{
        for (int i = 0; i < pointsNum; i++) {
            if(points[i*3] != points[0]){
                x_equal = false;
            }
            if(points[i*3+1] != points[1]) {
                y_equal = false;
            }
        }

        if(x_equal && y_equal){
            pre_outputs.push_back(".");
        }
        else if(x_equal || y_equal){
            if(x_equal){
                pre_outputs.push_back("|");
                pre_outputs.push_back("1");
                pre_outputs.push_back("I");
                pre_outputs.push_back("i");
                pre_outputs.push_back("l");
                pre_outputs.push_back("!");
            }
            if(y_equal) {
                pre_outputs.push_back("-");
                pre_outputs.push_back("一");
                pre_outputs.push_back("_");
                pre_outputs.push_back("--");
                pre_outputs.push_back("...");
            }
        }
    }

    return pre_outputs;
}

void TeaseID(float* featureData, int pointsNum, float* get_input_points) {
    float* new_s_points = new float[pointsNum]();

    for (int i = 0; i < pointsNum; i++) {
        new_s_points[i] = featureData[i * 3 + 2];
    }

    float s_min = get_array_min(new_s_points, pointsNum);
    float id_now = 0;
    float old_id = s_min;

    for(int i =0;i<pointsNum;i++){
        get_input_points[i*3+0] =  featureData[i*3+0];
        get_input_points[i*3+1] =  featureData[i*3+1];

        if(new_s_points[i] == old_id){
            new_s_points[i] = id_now;
        }
        else{
            old_id = new_s_points[i];
            id_now = id_now + 1;
            new_s_points[i] = id_now;
        }
        get_input_points[i*3+2] = new_s_points[i];
    }

    delete[] new_s_points;
}

vector<string> SortAndDeduplication(vector< Results > outputs){

    vector<string> string_outputs;
    map<float,string> map_conf2result;
    vector<float> overlap_outputs_confidence;

    string lm_result_str = "";

    overlap_outputs_confidence = outputs[0].m_lm_confidence;

    for(int i=0; i<outputs[0].m_lm_outputs.size(); i++){
        for(int j=0; j<outputs[0].m_lm_outputs[i].size(); j++){
            lm_result_str += dict_27769[outputs[0].m_lm_outputs[i][j]];
        }
        if(lm_result_str !=""){
            string_outputs.push_back(lm_result_str);
        }
        lm_result_str="";
    }

    for(int i=0; i<string_outputs.size(); i++){
        map_conf2result[overlap_outputs_confidence[i]] = string_outputs[i];
    }

    sort(overlap_outputs_confidence.begin(), overlap_outputs_confidence.end());
    reverse(overlap_outputs_confidence.begin(), overlap_outputs_confidence.end());

    for(int i=0; i<string_outputs.size(); i++){
        if(find(lm_result_array.begin(),lm_result_array.end(),map_conf2result[overlap_outputs_confidence[i]])==lm_result_array.end()){
            lm_result_array.push_back(map_conf2result[overlap_outputs_confidence[i]]);
            outputs_confidence.push_back(overlap_outputs_confidence[i]);
        }
        else{
            int index = find(lm_result_array.begin(),lm_result_array.end(),map_conf2result[overlap_outputs_confidence[i]]) - lm_result_array.begin();
            outputs_confidence[index] = outputs_confidence[index] + overlap_outputs_confidence[i];
        }

    }
    map_conf2result.clear();

    for(int i=0; i<lm_result_array.size(); i++){
        map_conf2result[outputs_confidence[i]] = lm_result_array[i];
    }

    sort(outputs_confidence.begin(), outputs_confidence.end());
    reverse(outputs_confidence.begin(), outputs_confidence.end());

    for(int i=0; i<lm_result_array.size(); i++){
        lm_result_array[i] = map_conf2result[outputs_confidence[i]];
    }

    vector<string>().swap(string_outputs);
    map<float,string>().swap(map_conf2result);
    vector<float>().swap(overlap_outputs_confidence);

    return lm_result_array;
}

// c++ test
bool OLHCT_Init()
{
    const int in_c = 6;
    const int d_model_1_ = 128;
    const int d_model_2_ = 512;
    const int n_lm_layer_ = 4;
    const int fc_svd_dims_ = 163;
    const int em_svd_dims_ = 155;
    const int transformer_svd_dims_ = 128;
    const int classes_ = 27769;

    Net.load_param(in_c, d_model_1_, d_model_2_, n_lm_layer_,fc_svd_dims_,em_svd_dims_,transformer_svd_dims_,classes_);
    AAsset* aAsset = new AAsset("/Users/xm210409/kika/olhct/openBlas_learn/assets/param.bin");
    Net.load_model(aAsset);
    Net.set();

    ifstream inf("/Users/xm210409/kika/olhct/openBlas_learn/assets/char_set_27769.txt");
    std::string s;
    dict_27769.clear();
    while(getline(inf, s)) {
        dict_27769.push_back(s);
    }

//    for (int i=0; i<dict_27769.size(); i++) {
//        std::cout << i +1 << " : [" << dict_27769[i] << "]" << endl;
//    }
    return true;
}

bool OLHCT_Release()
{
    OLHCT new_Net;
    vector<string>().swap(lm_result_array);
    vector<float>().swap(outputs_confidence);
    vector<string>().swap(dict_27769);
    Net = new_Net;
    return true;
}

vector<string> OLHCT_Recognize(float* points ,int pointsNum, int& featureSize)
{
    vector<float>().swap(outputs_confidence);
    vector<string>().swap(lm_result_array);

    float* get_input_points = new float[pointsNum*3];

    TeaseID(points, pointsNum, get_input_points);

    vector<string> dot_cross_vertical = PreRecognize(get_input_points, pointsNum);

    if(dot_cross_vertical.size() == 0){

        float* R2H_points = RotateToHorizontal(get_input_points, pointsNum);
        float* NPFL_points = NormalizePointsAdvanced(R2H_points, pointsNum, true);
        vector<float> SN_points = SpeedNormalize(NPFL_points, pointsNum, 0.2);
        featureSize = (pointsNum - 1)*6;
        float* LF_points = LineFeature(SN_points, pointsNum);

//        __android_log_print(ANDROID_LOG_DEBUG, "olhct", "%d ", pointsNum);

        vector< Results > outputs;

        outputs = Net.forward(LF_points, pointsNum);

        lm_result_array = SortAndDeduplication(outputs);
//        cout << "lm_result_array: " << lm_result_array.size() << endl;

        delete[] R2H_points;
        delete[] NPFL_points;
        vector<float>().swap(SN_points);
        delete[] LF_points;
        vector< Results >().swap(outputs);

    }
    else{
        lm_result_array = dot_cross_vertical;
        for(int i =0;i<lm_result_array.size();i++){
            outputs_confidence.push_back(1.0/lm_result_array.size());
        }
    }


    delete[] get_input_points;

    vector<string>().swap(dot_cross_vertical);

    vector<string> result = lm_result_array;

    vector<string>().swap(lm_result_array);
    return result;
}


class Eval {
public:
    Eval() {
        OLHCT_Init();
    }

    ~Eval() {
        OLHCT_Release();
    }

    void evalFile(std::string file, std::string saveTofile) {
        vector<WriteData> datas;
        cout << "loading..." << endl;
        Utils::loadFromFile(file, datas);
//        cout << WriteData::headString() << endl;
        uint64_t maxCost = 0;
        uint64_t minCost = 0;
        uint64_t sumCost = 0;

        cout << "Recognizing..." << endl;
        int count = 0;
        for (auto& data : datas) {
            Recognize(data);
            maxCost = std::max(uint64_t(data.cost), maxCost);
            minCost = std::min(uint64_t(data.cost), minCost);
            sumCost += uint64_t(data.cost);
            count++;
            if (count %1000 == 0) {
                cout << count << endl;
            }
        }

        cout << "max:\t" << maxCost << endl;
        cout << "min:\t" << minCost << endl;
        cout << "avg:\t" << sumCost*1.0/datas.size() << endl;
        cout << "saving..." << endl;
        Utils::saveToFile(saveTofile, datas);
        cout << "Done!!" << endl;
    }

private:
    void Recognize(WriteData& data) {
        auto start =  GetTimeInMs();
        data.results = OLHCT_Recognize(data.path.data(), data.path.size()/3, data.featureNum);
        data.result = data.results[0];
        data.props = outputs_confidence;
        data.cost = GetTimeInMs() - start;
    }

//    void saveTo(std::string saveTofile)
};



#endif //OPENBLAS_LEARN_NATIVE_H
