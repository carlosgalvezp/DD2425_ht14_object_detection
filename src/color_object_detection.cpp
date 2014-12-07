#include <object_detection/color_object_detection.h>
#include <object_detection/hsv_params.h>

#include <ras_utils/pcl_utils.h>

#include <vector>
Color_Object_Detection::Color_Object_Detection(const HSV_Filter &hsv_filter)
    :hsv_filter_(hsv_filter)
{
}

int Color_Object_Detection::detectColoredObject(const cv::Mat &bgr_img,
                                                 const cv::Mat &floor_mask,
                                                 cv::Mat &object_mask,
                                                 cv::Point &mass_center)
{
    // ** Apply mask to remove floor in 2D
    cv::Mat rgb_filtered;
    bgr_img.copyTo(rgb_filtered, floor_mask);

    // ** Color analysis
    cv::Mat color_mask;
    int color = color_analysis(rgb_filtered, mass_center, color_mask);

    // ** Combine color mask and floor mask
    cv::bitwise_and(color_mask, floor_mask, object_mask);

    return color;
}



int Color_Object_Detection::color_analysis(const cv::Mat &bgr_img,
                                           cv::Point2i &mass_center,
                                           cv::Mat &out_img)
{
    out_img = cv::Mat::zeros(bgr_img.rows, bgr_img.cols, CV_8UC1);

    // ** Parameters
    int erode_times=1;
    int dilate_times=1;

    // ** Convert to HSV
    cv::Mat img_hsv;
    cv::cvtColor(bgr_img,img_hsv,CV_BGR2HSV);

    // ** Create color masks
    std::vector<cv::Mat> img_binary;
    hsv_filter_.filter(img_hsv, img_binary);

    // ** Remove noise with erosion and dilation
    double max_pixel_count = 0;
    int max_idx = 0;
    for(std::size_t i = 0; i < img_binary.size(); ++i)
    {
        cv::dilate(img_binary[i],img_binary[i],cv::Mat(),cv::Point(-1,-1),dilate_times); //dilate image1 to fill empty spaces)
        cv::erode(img_binary[i],img_binary[i],cv::Mat(),cv::Point(-1,-1),erode_times); //erode image1 to remove single pixels
        double pixel_count = cv::countNonZero(img_binary[i]);
        if(pixel_count > max_pixel_count)
        {
            max_pixel_count = pixel_count;
            max_idx = i;
        }
    }

    // ** Select mask with biggest pixel count
    cv::Mat &mask = img_binary[max_idx];

    // ** Find biggest contour
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask,contours,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    if(contours.size() > 0)
    {
        std::vector<cv::Point> biggest_contour;
        // ** Get biggest contour
        double max_size = -1;
        double max_i = 0;
        for(std::size_t s = 0; s < contours.size(); ++s)
        {
            double size = cv::contourArea(contours[s]);
            if(size > max_size)
            {
                max_size = size;
                max_i = s;
            }
        }

        // ** Filter by aspect ratio and size
        cv::Rect bbox = cv::boundingRect(contours[max_i]);
        double aspect_ratio = std::max(bbox.height/bbox.width, bbox.width/bbox.height);

        if (max_size > MIN_PIXEL_OBJECT && aspect_ratio < MAX_ASPECT_RATIO)
        {
            //** Create mask
            biggest_contour = contours[max_i];
            cv::drawContours(out_img,contours, max_i, cv::Scalar(255), CV_FILLED);

            // ** Get mass center
            cv::Moments m = cv::moments(biggest_contour);
            mass_center.x = m.m10/m.m00;
            mass_center.y = m.m01/m.m00;

            return max_idx;
        }
    }
    return -1;
}



