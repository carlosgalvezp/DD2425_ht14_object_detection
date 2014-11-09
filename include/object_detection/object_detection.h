#include <iostream>
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

// Services
#include <ras_srv_msgs/Recognition.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define QUEUE_SIZE 2        // A too large value can cause more delay; it is preferred to drop frames

// ** Camera intrinsics (from /camera/depth_registered/camera_info topic)
#define SCALE_FACTOR    0.5
#define FX              574.0527954101562
#define FY              574.0527954101562
#define FX_INV          1.0/FX
#define FY_INV          1.0/FY
#define CX              319.5*SCALE_FACTOR
#define CY              239.5*SCALE_FACTOR
#define IMG_ROWS        480
#define IMG_COLS        640

// ** ROI (Region of Interest)
#define ROI_MIN_U       50              *SCALE_FACTOR
#define ROI_MAX_U       (IMG_COLS - 80) *SCALE_FACTOR
#define ROI_MIN_V       50              *SCALE_FACTOR
#define ROI_MAX_V       (IMG_ROWS - 50) *SCALE_FACTOR

// ** Object detection
#define MIN_MASS_CENTER_Y IMG_ROWS - 100
namespace object_detection
{
class Object_Detection{

    typedef image_transport::ImageTransport ImageTransport;
    typedef image_transport::Publisher ImagePublisher;
    typedef image_transport::SubscriberFilter ImageSubFilter;

    typedef message_filters::sync_policies::
        ApproximateTime<sensor_msgs::Image,
                        sensor_msgs::Image> RGBD_Sync_Policy;
    typedef message_filters::Synchronizer<RGBD_Sync_Policy> RGBD_Sync;

public:
    Object_Detection(const ros::NodeHandle& n, const ros::NodeHandle& n_private);
    ~Object_Detection(){}
    void RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                         const sensor_msgs::ImageConstPtr &depth_msg);

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
    void buildPointCloud(const cv::Mat &rgb_img,
                         const cv::Mat &depth_img,
                         pcl::PointCloud<pcl::PointXYZRGB> &cloud);
    void get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out);
    void backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                            cv::Mat &floor_mask);

    bool color_analysis(const cv::Mat &img, cv::Point2i &mass_center, cv::Mat &out_img);
    ros::WallTime ros_time;
    int frame_counter_;
    bool started_;
    bool detected_object_;
    cv::Mat ROI_;
};
}
