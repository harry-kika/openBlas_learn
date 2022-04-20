#include <iostream>
#include <vector>
#include "openblas/cblas.h"
#include "native.h"
#include <fstream>
#include "Utils.h"

using namespace std;

void RandomFill(vector<float>& numbers, size_t size) {
    srand((unsigned)time(NULL));
    numbers.resize(size);
    for (size_t i= 0; i<size; i++) {
        numbers[i] = float( rand() % 20);
    }
}

void Print(vector<float>& numbers, size_t row, size_t col, string name) {
    cout << "--- " << name << ":\t" << row << "*" << col << " ---"  << endl;
    for (size_t r = 0; r<row; r++) {
        for (size_t c = 0; c<col; c++) {
            cout << numbers[col*r + c] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

void TestLevel1() {

    vector<float> fv1 = {3, 5, 1, 0, 10, 2, 7};
    auto maxIndex = cblas_isamax(7, fv1.data(), 1);
    cout << "maxIndex:\t" << maxIndex << endl;
    cout << "maxValue:\t" << fv1[maxIndex] << endl;
}

void TestLevel2() {

    const int ROW = 3;
    const int COL = 2;

    vector<float> A = {1,2,3, 4,5,6};
    vector<float> x = {1, 1};
    vector<float> y = {0, 0};
//    RandomFill(A, ROW*COL);
    Print(A, ROW, COL, "A");

//    RandomFill(x, COL);
    Print(x, COL, 1, "x");

//    RandomFill(y, ROW);
    Print(y, 1, ROW, "y");

    /*
     *  实现 y := alpha * A * x + beta * y
     * */
    float alpha = 1.0;
    float beta = 2.0;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ROW, COL, alpha, A.data(), 2, x.data(), 1, beta, y.data(), 1);
    Print(y, 1, ROW, "r");
}



void TestFS() {
    float data[10];
    for (int i=0; i<10; i++) {
        data[i] = i/2.0;
    }
    for (int i=0; i<10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << endl;

    std::ofstream outF("testF.bin",std::ios::binary);
    outF.write(reinterpret_cast<char*>(data), sizeof(data));
    outF.close();

    std::ifstream  inF("testF.bin",std::ios::binary);
    float data1[3];
    float data2[7];
    inF.read(reinterpret_cast<char*>(data1), sizeof(float)*3);
    inF.read(reinterpret_cast<char*>(data2), sizeof(float)*7);
    inF.close();

    for (int i=0; i<3; i++) {
        std::cout << data1[i] << ",";
    }
    for (int i=0; i<7; i++) {
        std::cout << data2[i] << " ";
    }
    std::cout << endl;


    return;
}


void TEST_OLHCT() {
    auto start =  GetTimeInMs();
    cout << " init start" << endl;
    OLHCT_Init();
    cout << " init success: \t" << GetTimeInMs() - start << endl;

    vector<float> critic = {167.0, 79.0, 0.0, 156.0, 79.0, 0.0, 153.0, 79.0, 0.0, 138.0, 75.0, 0.0, 134.0, 75.0, 0.0, 123.0, 86.0, 0.0,
            118.0, 96.0, 0.0, 116.0, 102.0, 0.0, 110.0, 116.0, 0.0, 110.0, 122.0, 0.0, 110.0, 128.0, 0.0, 115.0, 131.0,
            0.0, 124.0, 140.0, 0.0, 132.0, 143.0, 0.0, 137.0, 143.0, 0.0, 139.0, 143.0, 0.0, 146.0, 143.0, 0.0, 151.0,
            143.0, 0.0, 154.0, 143.0, 0.0, 162.0, 137.0, 0.0, 164.0, 137.0, 0.0, 164.0, 137.0, 0.0, 91.0, 55.0, 1.0,
            110.0, 69.0, 1.0, 114.0, 74.0, 1.0, 115.0, 80.0, 1.0, 120.0, 100.0, 1.0, 121.0, 106.0, 1.0, 121.0, 119.0,
            1.0, 121.0, 125.0, 1.0, 121.0, 131.0, 1.0, 121.0, 135.0, 1.0, 121.0, 138.0, 1.0, 121.0, 140.0, 1.0, 121.0,
            141.0, 1.0, 121.0, 141.0, 1.0, 116.0, 88.0, 2.0, 123.0, 77.0, 2.0, 129.0, 71.0, 2.0, 134.0, 66.0, 2.0,
            136.0, 64.0, 2.0, 144.0, 57.0, 2.0, 148.0, 53.0, 2.0, 151.0, 51.0, 2.0, 153.0, 51.0, 2.0, 155.0, 50.0, 2.0,
            156.0, 50.0, 2.0, 158.0, 50.0, 2.0, 158.0, 50.0, 2.0, 158.0, 50.0, 2.0, 135.0, 78.0, 3.0, 135.0, 93.0, 3.0,
            135.0, 100.0, 3.0, 135.0, 107.0, 3.0, 135.0, 114.0, 3.0, 135.0, 118.0, 3.0, 132.0, 121.0, 3.0, 131.0, 125.0,
            3.0, 131.0, 126.0, 3.0, 130.0, 128.0, 3.0, 130.0, 128.0, 3.0, 129.0, 45.0, 4.0, 140.0, 52.0, 4.0, 140.0,
            52.0, 4.0, 137.0, 30.0, 5.0, 137.0, 47.0, 5.0, 132.0, 60.0, 5.0, 126.0, 92.0, 5.0, 124.0, 105.0, 5.0, 122.0,
            116.0, 5.0, 122.0, 122.0, 5.0, 122.0, 126.0, 5.0, 132.0, 126.0, 5.0, 138.0, 120.0, 5.0, 149.0, 112.0, 5.0,
            159.0, 107.0, 5.0, 161.0, 106.0, 5.0, 162.0, 103.0, 5.0, 162.0, 103.0, 5.0, 105.0, 69.0, 6.0, 119.0, 66.0,
            6.0, 129.0, 65.0, 6.0, 136.0, 65.0, 6.0, 138.0, 65.0, 6.0, 151.0, 62.0, 6.0, 151.0, 62.0, 6.0, 139.0, 76.0,
            7.0, 139.0, 93.0, 7.0, 139.0, 99.0, 7.0, 139.0, 109.0, 7.0, 139.0, 116.0, 7.0, 139.0, 120.0, 7.0, 139.0,
            125.0, 7.0, 139.0, 130.0, 7.0, 139.0, 133.0, 7.0, 139.0, 134.0, 7.0, 139.0, 134.0, 7.0, 139.0, 134.0, 7.0,
            132.0, 44.0, 8.0, 132.0, 44.0, 8.0, 136.0, 38.0, 9.0, 136.0, 38.0, 9.0, 141.0, 31.0, 10.0, 141.0, 39.0,
            10.0, 141.0, 43.0, 10.0, 141.0, 45.0, 10.0, 141.0, 45.0, 10.0, 147.0, 72.0, 11.0, 134.0, 72.0, 11.0, 126.0,
            72.0, 11.0, 121.0, 74.0, 11.0, 117.0, 79.0, 11.0, 112.0, 86.0, 11.0, 108.0, 96.0, 11.0, 107.0, 103.0, 11.0,
            107.0, 109.0, 11.0, 107.0, 118.0, 11.0, 111.0, 123.0, 11.0, 117.0, 130.0, 11.0, 127.0, 132.0, 11.0, 133.0,
            132.0, 11.0, 141.0, 129.0, 11.0, 148.0, 126.0, 11.0, 159.0, 121.0, 11.0, 159.0, 121.0, 11.0};

    vector<float> talk = {552, 143, 0, 552, 146, 0, 551, 157, 0, 552, 190, 0, 554, 225, 0, 559, 258, 0, 564, 269, 0, 568, 277, 0, 570, 278,
            0, 570, 275, 0, 563, 260, 0, 515, 221, 1, 517, 221, 1, 522, 221, 1, 536, 215, 1, 561, 206, 1, 585, 196, 1, 590,
            195, 1, 634, 201, 2, 633, 201, 2, 629, 202, 2, 625, 205, 2, 621, 211, 2, 617, 217, 2, 616, 223, 2, 618, 227, 2,
            622, 228, 2, 625, 225, 2, 629, 218, 2, 630, 209, 2, 631, 203, 2, 631, 198, 2, 633, 195, 2, 635, 198, 2, 637, 200,
            2, 643, 204, 2, 651, 207, 2, 659, 208, 2, 667, 204, 2, 674, 195, 2, 679, 184, 2, 683, 171, 2, 684, 154, 2, 685,
            142, 2, 685, 133, 2, 685, 128, 2, 683, 140, 2, 684, 163, 2, 692, 196, 2, 700, 219, 2, 707, 229, 2, 713, 234, 2,
            721, 235, 2, 722, 234, 2, 732, 103, 3, 732, 105, 3, 732, 117, 3, 732, 146, 3, 735, 185, 3, 741, 228, 3, 746, 248,
            3, 750, 259, 3, 757, 259, 3, 759, 257, 3, 776, 155, 4, 772, 159, 4, 765, 163, 4, 756, 170, 4, 746, 176, 4, 740,
            181, 4, 737, 184, 4, 743, 185, 4, 768, 182, 4, 798, 179, 4, 835, 174, 4, 863, 171, 4, 873, 170, 4, 879, 168, 4};

    start = GetTimeInMs();
    int featureSize = 0;
    vector<string> result = OLHCT_Recognize(talk.data(), talk.size()/3, featureSize);

//    vector<string> result = OLHCT_Recognize(critic.data(), critic.size()/3);
    cout << " OLHCT_Recognize success:\t" << GetTimeInMs() - start << endl;
    for (int i=0; i<result.size(); i++) {
        std::cout << i << " : " << result[i] << "\t" << outputs_confidence[i] << endl;
    }
    start = GetTimeInMs();
    OLHCT_Release();
    cout << " OLHCT_Release success:\t" << GetTimeInMs() - start << endl;
}


void test_eval() {
    Eval ev;
//    ev.evalFile("/Users/xm210409/kika/olhct/openBlas_learn/testData/test.txt",
//                "/Users/xm210409/kika/olhct/openBlas_learn/testData/test_result.txt");
    ev.evalFile("/Users/xm210409/kika/olhct/openBlas_learn/testData/single_0100_shuf_2w.txt",
                "/Users/xm210409/kika/olhct/openBlas_learn/testData/single_0100_shuf_2w_result.txt");
}




int main()
{
//    TestFS();
    TEST_OLHCT();
//    test_eval();


    return 0;
    std::cout << "Hello, openblas!" << std::endl;
    std::cout << "config:\t" << openblas_get_config() << std::endl;
    std::cout << "num_procs:\t" << openblas_get_num_procs() << std::endl;
    std::cout << "parallel:\t" << openblas_get_parallel() << std::endl;
    std::cout << "num_threads:\t" << openblas_get_num_threads() << std::endl;
    std::cout << "corename:\t" << openblas_get_corename() << std::endl;

//    TestLevel1();
//    TestLevel2();
    return 0;
}
