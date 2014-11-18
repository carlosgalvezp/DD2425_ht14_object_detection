#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <ras_utils/ras_utils.h>
// ROS
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <dynamic_reconfigure/server.h>
#include <object_detection/ColorTuningConfig.h>

// Services
#include <ras_srv_msgs/Recognition.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/LU>

#define QUEUE_SIZE 2        // A too large value can cause more delay; it is preferred to drop frames

// ** Camera intrinsics (from /camera/depth_registered/camera_info topic)
#define FX              574.0527954101562
#define FY              574.0527954101562
#define FX_INV          1.0/FX
#define FY_INV          1.0/FY
#define CX              319.5
#define CY              239.5
#define IMG_ROWS        480
#define IMG_COLS        640
#define SCALE_FACTOR    0.25

// ** ROI (Region of Interest)
#define ROI_MIN_U       50
#define ROI_MAX_U       (IMG_COLS - 80)
#define ROI_MIN_V       50
#define ROI_MAX_V       (IMG_ROWS - 50)

// ** Object detection
#define MIN_MASS_CENTER_Y IMG_ROWS - 100
namespace object_detection
{
class Object_Detection{

    struct HSV_Params
    {
        int R_H_min ,R_H_max, R_S_min, R_S_max, R_V_min, R_V_max;
        int G_H_min ,G_H_max, G_S_min, G_S_max, G_V_min, G_V_max;
        int B_H_min ,B_H_max, B_S_min, B_S_max, B_V_min, B_V_max;
        int Y_H_min ,Y_H_max, Y_S_min, Y_S_max, Y_V_min, Y_V_max;
        int P_H_min ,P_H_max, P_S_min, P_S_max, P_V_min, P_V_max;

        bool R_display, G_display, B_display, Y_display, P_display;
        bool remove_floor;
    };

    typedef message_filters::sync_policies::
        ApproximateTime<sensor_msgs::Image,
                        sensor_msgs::Image> RGBD_Sync_Policy;
    typedef message_filters::Synchronizer<RGBD_Sync_Policy> RGBD_Sync;


public:
    Object_Detection(const ros::NodeHandle& n, const ros::NodeHandle& n_private);
    ~Object_Detection(){}
    void RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                         const sensor_msgs::ImageConstPtr &depth_msg);
    void DR_Callback(object_detection::ColorTuningConfig &config, uint32_t level);

private:
    ros::NodeHandle n_, n_private_;
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;


    boost::shared_ptr<RGBD_Sync> rgbd_sync_;

    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;
    image_transport::Publisher mask_pub_;

    ros::ServiceClient service_client_;
    ros::Publisher speaker_pub_;

    dynamic_reconfigure::Server<object_detection::ColorTuningConfig> DR_server_;
    dynamic_reconfigure::Server<object_detection::ColorTuningConfig>::CallbackType DR_f_;

    ros::WallTime ros_time;
    int frame_counter_;
    bool started_;
    bool detected_object_;
    cv::Mat ROI_;

    HSV_Params hsv_params_;

    Eigen::Matrix4f t_cam_to_robot_;
    Eigen::Matrix4f t_robot_to_cam_;

    void buildPointCloud(const cv::Mat &rgb_img,
                         const cv::Mat &depth_img,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);
    void get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out);
    void backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                            cv::Mat &floor_mask);

    int color_analysis(const cv::Mat &img, cv::Point2i &mass_center, cv::Mat &out_img);

    void load_calibration(const std::string &path);
};
}
