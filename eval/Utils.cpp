//
// Created by Harry on 2022/4/19.
//

#include "Utils.h"
#include <fstream>
#include <iostream>

using namespace std;
namespace Utils {
    WriteData parseLine(const string& str){
        WriteData wd;
        bool isTextEnd(false);
        bool isFloatStart(false);
        string sf = "";
        for (auto& ch : str) {

            if (ch == '\t') {
                isTextEnd = true;
                continue;
            }

            if (ch == '[') {
                isFloatStart = true;
                continue;
            }

            if (ch == ',') {
                wd.path.emplace_back(stof(sf));
                sf = "";
                continue;
            }

            if (ch == ']') {
                break;
            }

            if (!isTextEnd) {
                wd.real += ch;
                continue;
            }

            if(isFloatStart) {
                sf += ch;
            }
        }

        wd.pointNum = wd.path.size();
        return wd;
    }

    void loadFromFile(std::string file, std::vector<WriteData>& datas)
    {
        std::cout << file << endl;
        ifstream inf(file);
        string str;
        while ( getline(inf, str) ) {
            if (str.empty()) {
                continue;
            }
            datas.emplace_back(parseLine(str));
        }
        return;
    }

    void saveToFile(std::string file, vector<WriteData>& datas)
    {
        ofstream outf(file);
        outf << WriteData::headString() << endl;
        for (auto& d : datas) {
            outf << d.toString() << endl;
        }
        outf.close();
    }

}


