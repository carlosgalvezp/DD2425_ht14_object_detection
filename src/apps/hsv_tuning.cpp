// STL
#include <fstream>
#include <iostream>

#include <object_detection/hsv_filter.h>
#include <object_detection/hsv_params.h>

// ROS
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// Utils
#include <ras_utils/types.h>
#include <ras_utils/basic_node.h>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <object_detection/ColorTuningConfig.h>


class HSV_Tuning : rob::BasicNode
{
public:
    HSV_Tuning(const std::string &filename);

private:
    void RGB_Callback(const sensor_msgs::ImageConstPtr &rgb_msg);
    void DR_Callback(object_detection::ColorTuningConfig &config, uint32_t level);

    void displayColorMasks(const std::vector<cv::Mat> &color_masks, const std::vector<bool> &display_mask);
    void saveHSVParams(const std::string &path, const std::vector<HSV_Params> &hsv_params);

    ros::Subscriber rgb_sub_;

    dynamic_reconfigure::Server<object_detection::ColorTuningConfig> DR_server_;
    dynamic_reconfigure::Server<object_detection::ColorTuningConfig>::CallbackType DR_f_;

    std::string filename_;
    std::vector<bool> display_color_masks_;
    std::vector<HSV_Params> hsv_params_;
    HSV_Filter hsv_filter_;
};


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./hsv_tuning <environment>" << std::endl;
        std::cerr << "Environment: " << std::endl;
        std::cerr << "0 - Lab" << std::endl;
        std::cerr << "1 - Atrium" << std::endl;
        return -1;
    }

    // ** Init node
    ros::init(argc, argv, "hsv_tuning");

    std::string path;
    int option = atoi(argv[1]);
    if(option)
        path = RAS_Names::HSV_PARAMS_LAB;
    else
        path = RAS_Names::HSV_PARAMS_CONTEST;

    // ** Create object
    HSV_Tuning obj(path);

    // ** Spin
    ros::spin();

    return 0;
}


// =====================================================================================
// =====================================================================================
HSV_Tuning::HSV_Tuning(const std::string &filename)
    : filename_(filename)
{
    // ** Subscribers
    rgb_sub_ = n.subscribe(TOPIC_CAMERA_RGB, 1,  &HSV_Tuning::RGB_Callback, this);

    // ** Dynamic reconfigure
    DR_f_ = boost::bind(&HSV_Tuning::DR_Callback, this, _1, _2);
    DR_server_.setCallback(DR_f_);
}

void HSV_Tuning::RGB_Callback(const sensor_msgs::ImageConstPtr &bgr_msg)
{
    // ** Convert ROS messages to OpenCV image and convert to HSV
    cv_bridge::CvImageConstPtr bgr_ptr = cv_bridge::toCvShare(bgr_msg);
    const cv::Mat& bgr_img = bgr_ptr->image;
    cv::Mat hsv_img;
    cv::cvtColor(bgr_img, hsv_img, CV_BGR2HSV);
    cv::imshow("BGR", bgr_img);
    cv::waitKey(1);

    // ** Apply HSV filtering using the HSV params from Dynamic reconfigure
    std::vector<cv::Mat> color_masks;
    hsv_filter_.filter(hsv_img, hsv_params_, color_masks);

    // ** Display masks if desired
    displayColorMasks(color_masks, display_color_masks_);
}

void HSV_Tuning::DR_Callback(object_detection::ColorTuningConfig &config, uint32_t level)
{
    ROS_INFO("[HSV_Tuning] Dynamic reconfigure");
    HSV_Params param_r(config.R_H_min, config.R_H_max, config.R_S_min, config.R_S_max, config.R_V_min, config.R_V_max);
    HSV_Params param_g(config.G_H_min, config.G_H_max, config.G_S_min, config.G_S_max, config.G_V_min, config.G_V_max);
    HSV_Params param_b(config.B_H_min, config.B_H_max, config.B_S_min, config.B_S_max, config.B_V_min, config.B_V_max);
    HSV_Params param_y(config.Y_H_min, config.Y_H_max, config.Y_S_min, config.Y_S_max, config.Y_V_min, config.Y_V_max);
    HSV_Params param_p(config.P_H_min, config.P_H_max, config.P_S_min, config.P_S_max, config.P_V_min, config.P_V_max);

    hsv_params_ = {param_r, param_g, param_b, param_y, param_p};
    display_color_masks_ = {config.R_Display, config.G_Display, config.B_Display, config.Y_Display, config.P_Display};

    this->saveHSVParams(this->filename_, this->hsv_params_);
}

void HSV_Tuning::displayColorMasks(const std::vector<cv::Mat> &color_masks, const std::vector<bool> &display_mask)
{
    for(std::size_t i = 0; i < color_masks.size(); ++i)
    {
        if(display_mask[i])
        {
            std::stringstream ss;
            ss << "Mask "<<i;
            cv::imshow(ss.str().c_str(), color_masks[i]);
        }
    }
}

void HSV_Tuning::saveHSVParams(const std::string &path, const std::vector<HSV_Params> &hsv_params)
{
    std::ofstream file(path);

    file << hsv_params.size() << std::endl;

    for(std::size_t i = 0; i < hsv_params.size(); ++i)
    {
        file << hsv_params[i] << std::endl;
    }
    file.close();
}
