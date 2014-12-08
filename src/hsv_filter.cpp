#include <object_detection/hsv_filter.h>

HSV_Filter::HSV_Filter()
{
}

HSV_Filter::HSV_Filter(const std::string &hsv_params_path)
{
    std::cout << "HSV FILTER "<<hsv_params_path<<std::endl;
    this->readHSVParams(hsv_params_path, this->hsv_params_);
}

void HSV_Filter::filter(const cv::Mat &hsv_img, const HSV_Params &hsv_params, cv::Mat &color_mask)
{
    cv::inRange(hsv_img, cv::Scalar(hsv_params.H_min, hsv_params.S_min, hsv_params.V_min),
                         cv::Scalar(hsv_params.H_max, hsv_params.S_max, hsv_params.V_max),
                         color_mask);
}

void HSV_Filter::filter(const cv::Mat &hsv_img, const std::vector<HSV_Params> &hsv_params, std::vector<cv::Mat> &color_masks)
{
    color_masks.resize(hsv_params.size());
    for(std::size_t i = 0; i < hsv_params.size(); ++i)
    {
        this->filter(hsv_img, hsv_params[i], color_masks[i]);
    }
}

void HSV_Filter::filter(const cv::Mat &hsv_img, std::vector<cv::Mat> &color_masks)
{
    this->filter(hsv_img, this->hsv_params_, color_masks);
}

void HSV_Filter::readHSVParams(const std::string &path, std::vector<HSV_Params> &hsv_params)
{
    std::ifstream file(path);
    int n;
    file >> n;
    hsv_params.resize(n);

    for (std::size_t i = 0; i < hsv_params.size(); ++i)
    {
        int h_min, h_max, s_min, s_max, v_min, v_max;
        file >> h_min >> h_max >> s_min >> s_max >> v_min >> v_max;
        HSV_Params p(h_min, h_max, s_min, s_max, v_min, v_max);

        hsv_params[i] = p;
    }
}
