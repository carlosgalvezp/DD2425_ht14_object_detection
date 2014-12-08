#ifndef HSV_FILTER_H
#define HSV_FILTER_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <object_detection/hsv_params.h>

#include <vector>
#include <fstream>
#include <iostream>

class HSV_Filter
{
public:
    HSV_Filter();
    HSV_Filter(const std::string &hsv_params_path);

    void filter(const cv::Mat &hsv_img, const HSV_Params &hsv_params, cv::Mat &color_mask);
    void filter(const cv::Mat &hsv_img, const std::vector<HSV_Params> &hsv_params, std::vector<cv::Mat> &color_masks);
    void filter(const cv::Mat &hsv_img, std::vector<cv::Mat> &color_masks);

private:
    std::vector<HSV_Params> hsv_params_;

    void readHSVParams(const std::string &path, std::vector<HSV_Params> &hsv_params);
};

#endif // HSV_FILTERING_H
