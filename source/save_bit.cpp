/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */
#include "save_bit.h"
//#include <android/log.h>

#include <iostream>
#include "ReadBits.h"
#include "CThreadPool.h"
#include "math.h"

void ReadOneBit(byte *pBuffer, int nStart, /* out */int &nEnd, /* out */ byte &retByte)
{
    byte btData = pBuffer[nStart / 8];
    btData = btData << nStart % 8;
    retByte = btData >> 7;
    nEnd = nStart + 1;
}

void ReadStringFromBuffer(byte *pBuffer, int nStart, int nCount, /* out */int &nEnd, /* out */
                          char *pRetData)
{
    for (int nIndex = 0; nIndex < nCount; nIndex++) {
        ReadDataFromBuffer(pBuffer, nStart, 8, nStart, pRetData[nIndex]);
    }
    nEnd = nStart;
}


void WriteOneBit(byte *pBuffer, byte btData, int nStart,  /* out */int &nEnd)
{
    int nSet = nStart / 8;
    byte c = pBuffer[nSet];
    switch (btData) {
        case 1:
            c |= (1 << (7 - nStart % 8));
            break;
        case 0:
            c &= (~(1 << (7 - nStart % 8)));
            break;
        default:
            return;
    }
    pBuffer[nSet] = c;
    nEnd = nStart + 1;
}

void WtriteStringToBuffer(byte *pBuffer, char *pchar, int nStart, int nCount, /* out */int &nEnd)
{
    for (int nIndex = 0; nIndex < nCount; nIndex++) {
        WriteDataToBuffer(pBuffer, pchar[nIndex], nStart, 8, nStart);
    }
    nEnd = nStart;
}


void CHECK_IF(const bool a)
{
    if (!a) {
        printf("Open File Error!\n");
        assert(0);
    }
}

void CHECK_INFO(const bool a, const std::string info)
{
    if (!a) {
        printf("%s\n", info.c_str());
        assert(0);
    }
}

const int get_log_two(const int data)
{
    if (data == 0)
        printf("cluster_center is error\n");
    int count = 0, count_data = 1;
    while (data > count_data) {
        count_data *= 2;
        count++;
    }
    return count;
}

void reconvery_blob_data(float *weights, const float *cluster_center,
                         const int *array_cluster_label_index, const int diff_index_pair_length, \
  const int weights_num, const byte *array_diff)
{

    int count_blob_index = 0, index_pair = 0;
    count_blob_index = array_diff[0];
    weights[count_blob_index] = cluster_center[array_cluster_label_index[0]];
    index_pair++;

    for (; index_pair < diff_index_pair_length; ++index_pair) {
        count_blob_index += (array_diff[index_pair] + 1);
        weights[count_blob_index] = cluster_center[array_cluster_label_index[index_pair]];

    }
}

bool unpack_cluster_blob(float *weights, AAsset *&fp, int weightSize)
{
    int count_weight_num;
    int save_diff_index_byte_size;
    int index_bit;
    byte *save_byte;
    byte *array_diff;
    int *array_cluster_label_index;
    int diff_index_pair_length;
    int fread_temp = 0;
    int compress_byte = 0;

    unsigned short count_cluster_center;
    fread_temp = AAsset_read(fp, &count_cluster_center, sizeof(unsigned short) * 1);
//  fread_temp = fread(&count_cluster_center, sizeof(unsigned short),1,fp);
    const int count_cluster_center_const = count_cluster_center;
    compress_byte += 2; // 2 bytes

    float *cluster_center = new float[count_cluster_center_const]();

    fread_temp = AAsset_read(fp, cluster_center, sizeof(float) * count_cluster_center_const);
//  fread_temp = fread(cluster_center, sizeof(float),count_cluster_center_const,fp);


    compress_byte += 4 * count_cluster_center_const; // 4bytes

    const int weight_bit = get_log_two(count_cluster_center);

    fread_temp = AAsset_read(fp, &count_weight_num, sizeof(int) * 1);
//  fread_temp = fread(&count_weight_num, sizeof(int),1,fp);

//    __android_log_print(ANDROID_LOG_DEBUG, "olhct", "%d ", count_weight_num);
//    __android_log_print(ANDROID_LOG_DEBUG, "olhct", "%d ", weightSize);
    // cout <<  "count_weight_num = " << count_weight_num;
    // cout <<  "weightSize = " << weightSize;

    CHECK_INFO(count_weight_num == weightSize, "don't match");
    compress_byte += 4;

    fread_temp = AAsset_read(fp, &save_diff_index_byte_size, sizeof(int) * 1);
//  fread_temp = fread(&save_diff_index_byte_size, sizeof(int),1,fp);
    fread_temp = AAsset_read(fp, &index_bit, sizeof(unsigned char) * 1);
//  fread_temp = fread(&index_bit, sizeof(unsigned char),1,fp);
    fread_temp = AAsset_read(fp, &diff_index_pair_length, sizeof(int) * 1);
//  fread_temp = fread(&diff_index_pair_length, sizeof(int),1,fp);
    const int diff_index_pair_length_const = diff_index_pair_length;
    compress_byte += 4 + 1 + 4;

    array_diff = new byte[diff_index_pair_length_const]();
    array_cluster_label_index = new int[diff_index_pair_length_const]();

    save_byte = new byte[save_diff_index_byte_size]();
    fread_temp = AAsset_read(fp, save_byte, sizeof(byte) * save_diff_index_byte_size);
//  fread_temp=fread(save_byte, sizeof(byte),save_diff_index_byte_size,fp);
    compress_byte += save_diff_index_byte_size;

//    uint32 nStart = 0;
//    for (int i = 0; i < diff_index_pair_length_const; i++) {
//        nStart = ReadBits(save_byte, nStart, index_bit, array_diff[i]);
//        nStart = ReadBits(save_byte, nStart, weight_bit, array_cluster_label_index[i]);
//    }

typedef struct st_param{
    byte* buf;
    uint32 bufLen;
    uint32 elemIndex;
    uint32 elemLen;
    uint32 bitLen1;
    uint32 bitLen2;
    byte *array_1;
    int *array_2;

    st_param(byte *buf, uint32 bufLen, uint32 elemIndex, uint32 elemLen, uint32 bitLen1, uint32 bitLen2,
             byte *array1, int *array2) : buf(buf), bufLen(bufLen), elemIndex(elemIndex),
                                          elemLen(elemLen), bitLen1(bitLen1), bitLen2(bitLen2),
                                          array_1(array1), array_2(array2)
    {}
}st_param;

auto func = [](void* arg) {
    st_param * param = (st_param*)arg;
    byte* buf = param->buf;
    uint32 bufLen = param->bufLen;
    uint32 elemIndex = param->elemIndex;
    uint32 elemLen = param->elemLen;
    uint32 bitLen1 = param->bitLen1;
    uint32 bitLen2 = param->bitLen2;
    byte *array_1 = param->array_1;
    int *array_2 = param->array_2;
    delete param;

    uint32 gsbit = (bitLen1 + bitLen2) * (elemIndex);
    for(uint32 i = elemIndex; i<bufLen && i<elemIndex + elemLen; ++i) {
        gsbit = ReadBitsBy4(buf, gsbit, bitLen1, array_1[i]);
        gsbit = ReadBitsBy4(buf, gsbit, bitLen2, array_2[i]);
    }
};

    index_bit = (byte)index_bit;
    int poolSize = CThreadPool::GetInstance().GetPoolSize();
    int elemlen = ceil(diff_index_pair_length_const / poolSize);
    for(int i = 0; i<diff_index_pair_length_const; i+=elemlen) {
        CThreadPool::GetInstance().AddTask(func, new st_param(save_byte, diff_index_pair_length_const, i, elemlen, index_bit, weight_bit, array_diff, array_cluster_label_index));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CThreadPool::GetInstance().Start();
    CThreadPool::GetInstance().Wait();

//    uint32 nStart = 0;
//    for (int i = 0; i < diff_index_pair_length_const; i++) {
//        nStart = ReadBitsBy4(save_byte, nStart, index_bit, array_diff[i]);
//        nStart = ReadBitsBy4(save_byte, nStart, weight_bit, array_cluster_label_index[i]);
//    }

    reconvery_blob_data(weights, cluster_center, array_cluster_label_index, diff_index_pair_length,
                        count_weight_num, array_diff);

    delete[] array_diff;
    delete[] array_cluster_label_index;
    delete[] save_byte;
    return true;
}