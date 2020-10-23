#include "dataformats.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

std::string split(const std::string &str, char delim, size_t &);
int readInt(const std::string &str, size_t &);
float readFloat(const std::string &str, size_t &);
std::pair<ModuleID, bool> parseModuleID(const std::string &);
StsDigi parseDigi(const std::string &);
StsCluster parseCluster(const std::string &);

std::ostream &operator<<(std::ostream &os, const StsDigi &digi) {
    os << "StsDigi{ " 
       << ".charge = " << digi.charge << ", "
       << ".channel = " << digi.channel << ", "
       << ".time = " << digi.time
       << " }";
    
    return os;
}

std::ostream &operator<<(std::ostream &os, const StsCluster &cls) {
    os << "StsCluster{ "
       << ".size = " << cls.size << ", "
       << ".charge = " << cls.charge << ", "
       << ".position = " << cls.position << ", "
       << ".positionError = " << cls.positionError << ", "
       << ".time = " << cls.time << ", "
       << ".timeError = " << cls.timeError
       << " }";

    return os;
}

OwningStsContainer<StsDigi> readDigis(const std::string &fname) {
    std::ifstream file{fname};
    assert(file.good());

    std::vector<StsDigi> digis;
    std::vector<size_t> offsets;

    std::pair<ModuleID, bool> mod{}; // module ID + isfront
    for (std::string line; std::getline(file, line);) {
        if (line.rfind("Module", 0) == 0) {
            mod = parseModuleID(line);
            offsets.push_back(digis.size());
        } else {
            StsDigi digi = parseDigi(line);
            digis.push_back(digi);
        }
    }

    offsets.push_back(digis.size());

    return OwningStsContainer<StsDigi>{offsets, digis};
}

// TODO: parse Cluster

std::string split(const std::string &str, char delim, size_t &start) {
    size_t end = str.find(delim, start);
    
    if (end == std::string::npos) {
        size_t oldStart = start;
        start = std::string::npos;
        return str.substr(oldStart);
    }

    std::string token = str.substr(start, end-start);
    start = end + 1;
    return token;
}

int readInt(const std::string &str, size_t &pos) {
    size_t len;
    int r = std::stoi(&str[pos], &len);
    pos += len + 1;
    return r;
}

float readFloat(const std::string &str, size_t &pos) {
    size_t len;
    float r = std::stof(&str[pos], &len);
    pos += len + 1;
    return r;
}

std::pair<ModuleID, bool> parseModuleID(const std::string &str) {
    std::pair<ModuleID, bool> res;

    size_t pos = 0;
    split(str, ' ', pos); // Skip Module
    res.first = std::stoi(split(str, ' ', pos));
    res.second = (split(str, ' ', pos) == "front");

    assert(pos == std::string::npos);

    return res;
}

StsDigi parseDigi(const std::string &str) {
    StsDigi digi;
    size_t pos = 0;

    readInt(str, pos); // Skip address
    digi.channel = readInt(str, pos);
    digi.charge = readFloat(str, pos);
    digi.time = readInt(str, pos);

    assert(pos == str.size() + 1);

    return digi;
}

StsCluster parseCluster(const std::string &str) {
    StsCluster cluster;
    size_t pos = 0;

    // Use readFloat / readInt instead.
    cluster.charge = std::stof(split(str, ';', pos));
    split(str, ';', pos); // skip index
    cluster.position = std::stof(split(str, ';', pos));
    cluster.positionError = std::stof(split(str, ';', pos));
    cluster.size = std::stoi(split(str, ';', pos));
    cluster.time = std::stof(split(str, ';', pos));
    cluster.timeError = std::stof(split(str, ';', pos));

    assert(pos == std::string::npos);

    return cluster;
}