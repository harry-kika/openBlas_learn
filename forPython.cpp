//
// Created by Harry on 2022/4/14.
//

#include "iostream"
#include "native.h"
#include "math.h"

void TEST_NATIVE() {

}

using namespace std;
extern "C" {
    void testP() {
        vector<float> fv1 = {3, 5, 1, 0, 10, 2, 7};
        auto maxIndex = cblas_isamax(7, fv1.data(), 1);
        cout << "maxIndex:\t" << maxIndex << endl;
        cout << "maxValue:\t" << fv1[maxIndex] << endl;
        string s = "test sucess";
        std::cout << s << endl;
    }

    void CTeaseID(float* featureData, int pointsNum, float* get_input_points) {
        TeaseID(featureData, pointsNum, get_input_points);
//        for (int i =0; i< pointsNum*3; i++) {
//            get_input_points[i] = featureData[i];
//        }
    }

    void CRotateToHorizontal(float* get_input_points, int pointsNum, float* out_points) {
        auto out = RotateToHorizontal(get_input_points, pointsNum);
        for (int i=0; i<pointsNum*3; i++) {
            out_points[i] = out[i];
        }
    }

    int CSpeedNormalize(float* points, int pointsNum, float density, float* out_points) {
        auto out = SpeedNormalize(points, pointsNum, density);
        for (int i=0; i<pointsNum*3; i++) {
            out_points[i] = out[i];
        }
        return pointsNum;
    }

    int COLHCT_Init() {
        OLHCT_Init();
        return 0;
    }

    int COLHCT_Release() {
        OLHCT_Release();
        return 0;
    }

    int COLHCT_Recognize(float* points ,int pointsNum, char* buf, int buffSize) {
        cout <<"---" << pointsNum << endl;
        auto start = GetTimeInMs();
        auto result = OLHCT_Recognize(points, pointsNum);
        int cost = GetTimeInMs() - start;

        cout <<"cost(ms)" << cost << endl;
        if (result.size() == 0) {
            OLHCT_Release();
            return cost;
        }
        int l = buffSize;
        result[0] = "makr";
        if (result[0].size() < l)
        {
            l = result[0].size();
        }

        for (int i=0; i<l; i++) {
            buf[i] = result[0][i];
        }
        std::cout << result[0]<< ":" << buf << endl;

        for (auto r : result) {
            std::cout << r << endl;
        }
        return cost;
    }
}



