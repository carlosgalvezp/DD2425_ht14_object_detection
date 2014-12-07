#ifndef COLOR_OBJECT_DETECTION_H
#define COLOR_OBJECT_DETECTION_H

#include <object_detection/hsv_filter.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_types.h>


// Image parameters
#define MIN_PIXEL_OBJECT        200         // [pixel] Min number of pixels to classify blob as object
#define MAX_ASPECT_RATIO        2           // Aspect ratio  = max(h/w, w/h), w,h are the dimensions of the blob's bounding box

class Color_Object_Detection
{
public:
    Color_Object_Detection(const HSV_Filter &hsv_filter);

    int detectColoredObject(const cv::Mat &bgr_img, const cv::Mat &floor_mask,
                            cv::Mat &object_mask, cv::Point &mass_center);

private:
    HSV_Filter hsv_filter_;

    int color_analysis(const cv::Mat &bgr_img,
                       cv::Point2i &mass_center,
                       cv::Mat &out_img);
    double estimateDepth(const cv::Mat &depth_img, cv::Point mass_center);

};

#endif // COLOR_OBJECT_DETECTION_H
