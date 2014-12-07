#ifndef HSV_PARAMS_H
#define HSV_PARAMS_H

#include <vector>
#include <ostream>

struct HSV_Params
{
    int H_min;
    int H_max;
    int S_min;
    int S_max;
    int V_min;
    int V_max;

    HSV_Params(){}
    HSV_Params(int h_min, int h_max, int s_min, int s_max, int v_min, int v_max)
        :H_min(h_min),
         H_max(h_max),
         S_min(s_min),
         S_max(s_max),
         V_min(v_min),
         V_max(v_max)
    {}
};

std::ostream& operator<<(std::ostream& os, const HSV_Params &params);

#endif // HSV_PARAMS_H
