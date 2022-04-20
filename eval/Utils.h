//
// Created by Harry on 2022/4/19.
//

#ifndef OPENBLAS_LEARN_UTILS_H
#define OPENBLAS_LEARN_UTILS_H

#include <string>
#include <vector>


struct WriteData {
    std::string real = "";
    std::vector<float> path;
    int pointNum = 0;
    int featureNum = 0;

    int cost = 0;
    std::string result = "";
    std::vector<std::string> results;
    std::vector<float> props;

    static std::string headString() {
        return "#real\tresult\tcost(ms)\tpointNum\tfeatureSize\tAllResult\tprops";
    }

    std::string toString() {
        std::string str = real;
        std::string tab = "\t";
        str += tab + result;
        str += tab + std::to_string(cost);
        str += tab + std::to_string(pointNum);
        str += tab + std::to_string(featureNum);
        str += tab + "[";
        for (auto& r:results) {
            str.append(r + ",");
        }
        str.erase(str.size()-1, 1);
        str.append("]");

        str.append(tab + "[");
        for (auto& p:props) {
            str.append(std::to_string(p) + ",");
        }
        str.erase(str.size()-1, 1);
        str.append("]");
        return str;
    }
};

namespace Utils {
    void loadFromFile(std::string file, std::vector<WriteData>& datas);

    void saveToFile(std::string file, std::vector<WriteData>& datas);
}


#endif //OPENBLAS_LEARN_UTILS_H
