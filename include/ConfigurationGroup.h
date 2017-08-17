#ifndef CONFIGURATION_GROUP_H
#define CONFIGURATION_GROUP_H

#include <iostream>
#include <map>
#include <string>

struct configurationGroup : std::map <std::string, std::string> {
    bool iskey(const std::string& s) const {
        return count(s) != 0;
    }
};
std::istream& operator >> ( std::istream& ins, configurationGroup& d );

#endif