//
// Created by xm210408 on 2022/3/30.
//

#ifndef OLHCT2SOLIBRARY_READBITS_H
#define OLHCT2SOLIBRARY_READBITS_H

using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;

using byte = uint8;

namespace {
    const uint32 BIT_COUNT_TO_BYTE4 = 25;
    const uint32 BIT_COUNT_TO_BYTE2 = 9;
}

//大端转小端
uint16 big2little(uint16 value)
{
    return (((value >> 8) & 0x00FF) | ((value << 8) & 0xFF00));
}

uint32 big2little(uint32 value)
{
    return ((value >> 24) & 0xff)
           | ((value >> 8) & 0xFF00)
           | ((value << 8) & 0xFF0000)
           | ((value << 24));
}

int16 big2little(int16 value)
{
    return (((value >> 8) & 0x00FF) | ((value << 8) & 0xFF00));
}

int32 big2little(int32 value)
{
    return ((value >> 24) & 0xff)
           | ((value >> 8) & 0xFF00)
           | ((value << 8) & 0xFF0000)
           | ((value << 24));
}

uint32 CalcByteLength(uint32 bits)
{
    if (bits == 0) {
        return 0;
    }
    int p = bits / 8;
    int q = bits % 8;
    return p + (q > 1 ? 2 : 1);
}

/*!
 * 调用之前，由调用者保证：
 *      1. buf有效，且buf[byteOff]不越界
 *      2. bit位区间[gsbit, gsbit + bitCount) 应在一个字节内
 */

byte ReadBitsInOne(const byte &src, byte sbit, byte bitCount)
{
    byte tmp = src << sbit;
    return tmp >> (8 - bitCount);
}

// 任意字节
template<typename T>
uint32 ReadBits(byte *buf, uint32 gsbit, uint32 bitCount, T &data)
{
    // 获取起始bit和结尾bit所在的字节索引
    uint32 gsbyte = gsbit / 8;
    uint32 gebyte = (gsbit + bitCount - 1) / 8;

    // 起始bit在所在字节内的索引[0, 8)
    uint32 sbit = gsbit % 8;
    if (gsbyte == gebyte) { // 处于同一个字节索引
        data = ReadBitsInOne(buf[gsbyte], sbit, bitCount);
    } else { // 起始bit和结尾bit不在同一个字节索引中，属于跨字节
        // 结尾bit在所在字节内的索引[0, 8)
        uint32 ebit = (gsbit + bitCount) % 8; // 索引区间，左闭右开，[gsbit, gsbit + bitCount)

        data = ReadBitsInOne(buf[gsbyte++], sbit, 8 - sbit);
        while (gsbyte < gebyte) {
            data = (data << 8) | buf[gsbyte++];
        }
        data = (data << ebit) | ReadBitsInOne(buf[gebyte], 0, ebit);
    }

    return gsbit + bitCount;
}

/*!
 *  调用前确保：
 *      1.确保buf有效，以及buf的长度不越界，如果buf越界，请使用ReadDataFromBuffer()函数。
 *      2.从gsbit开始的bitCount个bit都在同一个字节内
 */
template<typename T>
uint32 ReadBitsBy1(byte *buf, uint32 gsbit, uint32 bitCount, T &data)
{
    int byteOff = gsbit / 8;
    int bitOff = gsbit % 8;
    byte byte1 = (buf[byteOff] << bitOff) >> (8 - bitCount);
    data = byte1;
    return gsbit + bitCount;
}

/* 按4字节跨度来从buf读取bitCount个bit位数据
 * ReadBits使用说明，为了提高效率，函数内不做如下判断，需要在调用之间加以判断。
 *      1.确保类型T为4字节
 *      2.确保buf有效，以及buf的长度不越界，如果buf越界，请使用ReadDataFromBuffer()函数。
 *      3.确保bitCount不大于25。函数内按四字节读取，基于CalcByteLength()，四字节对应的bitCount范围在[17, 25]之间，大于25，
 *        此时bitCount的跨度可能大于4个字节。所以超过25bit不适用。
 */
template<typename T>
uint32 ReadBitsBy4(byte *buf, uint32 gsbit, uint32 bitCount, T &data)
{
    int byteOff = gsbit / 8;
    int bitOff = gsbit % 8;
//    uint32 src = *(uint32 *) (buf + byteOff);
//    src = big2little(src);
//    data = (src << bitOff) >> (sizeof(uint32) * 8 - bitCount);

    data = ((big2little(*(uint32 *) (buf + byteOff)) << bitOff) >> (32 - bitCount));
    return gsbit + bitCount;
}

/* 按4字节跨度来从buf读取bitCount个bit位数据，bitCount大于25时，每次读取25bit
 *
 * @tparam T :参数类型
 * @param buf : 待读取的buf
 * @param bitPos: 要读取的bit下标位置
 * @param bitCount: 要读取的bit长度
 * @param data : 保存读取的结果
 * @return : 返回下一次读取bit的下标位置
 *
 * ReadBits使用说明，为了提高效率，函数内不做如下判断，需要在调用之间加以判断。
 *      1.调用之前，先确保buf有效，以及buf的长度不越界，如果buf越界，请使用ReadDataFromBuffer()函数。
 *      2.没有bitcount长度限制，内部采用循环读取25bit的方式
 */
template<typename T>
uint32 ReadBitsBy4Ex(byte *buf, uint32 gsbit, uint32 bitCount, T &data)
{
    data = 0;
    uint32 tmpdata = 0;
    while (bitCount > BIT_COUNT_TO_BYTE4) {
        gsbit = ReadBitsBy4(buf, gsbit, BIT_COUNT_TO_BYTE4, tmpdata);
        data |= tmpdata << (bitCount - BIT_COUNT_TO_BYTE4);
        bitCount -= BIT_COUNT_TO_BYTE4;
    }
    gsbit = ReadBitsBy4(buf, gsbit, bitCount, tmpdata);
    data |= tmpdata;

    return gsbit;
}

#endif //OLHCT2SOLIBRARY_READBITS_H
