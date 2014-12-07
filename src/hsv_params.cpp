#include <object_detection/hsv_params.h>

std::ostream& operator<<(std::ostream& os, const HSV_Params &params)
{
    os << params.H_min << " "
       << params.H_max << " "
       << params.S_min << " "
       << params.S_max << " "
       << params.V_min << " "
       << params.V_max;
}

