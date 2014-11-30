#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>
#include <ras_utils/ras_utils.h>
#include <ras_utils/basic_node.h>
#include <ras_utils/pcl_utils.h>
#include <ras_utils/ras_names.h>
#include <object_recognition/object_recognition.h>
#include <ras_msgs/RAS_Evidence.h>

// ROS
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Imu.h"
#include <geometry_msgs/Pose2D.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

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
#define SCALE_FACTOR    0.25

// ** ROI (Region of Interest)
#define ROI_MIN_U       50
#define ROI_MAX_U       (IMG_COLS)- 80
#define ROI_MIN_V       50
#define ROI_MAX_V       (IMG_ROWS - 50)

// ** Object detection
// Image parameters
#define MIN_PIXEL_OBJECT        200                       // [pixel] Min number of pixels to classify blob as object
#define MAX_ASPECT_RATIO        10 // 1.5                       // Aspect ratio  = max(h/w, w/h), w,h are the dimensions of the blob's bounding box

// Geometric parameters
#define ROBOT_BORDER            0.115                     // [m] Distance from camera to robot front border
#define D_OBJECT_DETECTION      ROBOT_BORDER + 0.21       // [m] Distance at which we start trying to detect the object
#define D_OBEJCT_STOP           ROBOT_BORDER + 0.09       // [m] Distance at which we stop in front of the object
#define MIN_DIST_OBSTACLE       ROBOT_BORDER + 0.08       // [m] Distance at which we stop in front of an obstacle (wall)
#define NEW_OBJECT_MIN_DISTANCE 0.2                       // [m] Min distance between objects


#define CROP_SIZE   200 // Size of the square placed at the mass center of the object, cropping the image for further processing
namespace object_detection
{
class Object_Detection : rob::BasicNode{

    struct HSV_Params
    {
        std::vector<int> H_min;
        std::vector<int> H_max;
        std::vector<int> S_min;
        std::vector<int> S_max;
        std::vector<int> V_min;
        std::vector<int> V_max;

        std::vector<bool> display;
        bool remove_floor;
    };

    struct Object
    {
        pcl::PointXY position_;
        std::string id_;

        Object(pcl::PointXY position, const std::string &id)
            :position_(position), id_(id){}
    };

    typedef message_filters::sync_policies::
        ApproximateTime<sensor_msgs::Image,
                        sensor_msgs::Image> RGBD_Sync_Policy;
    typedef message_filters::Synchronizer<RGBD_Sync_Policy> RGBD_Sync;


public:
    Object_Detection();
    ~Object_Detection(){}
    void RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                         const sensor_msgs::ImageConstPtr &depth_msg);
    void DR_Callback(object_detection::ColorTuningConfig &config, uint32_t level);

    void IMUCallback(const sensor_msgs::ImuConstPtr &imu_msg);
    void OdometryCallback(const geometry_msgs::Pose2DConstPtr &odometry_msg);

private:
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;

    boost::shared_ptr<RGBD_Sync> rgbd_sync_;

    ros::ServiceClient service_client_;

    ros::Publisher speaker_pub_, evidence_pub_, obstacle_pub_, marker_pub_;
    ros::Publisher pcl_pub_, object_pcl_pub_;
    ros::Subscriber imu_sub_, odometry_sub_;

    dynamic_reconfigure::Server<object_detection::ColorTuningConfig> DR_server_;
    dynamic_reconfigure::Server<object_detection::ColorTuningConfig>::CallbackType DR_f_;

    ros::WallTime ros_time;
    int frame_counter_;
    cv::Mat ROI_;

    HSV_Params hsv_params_;

    Eigen::Matrix4f t_cam_to_robot_;
    Eigen::Matrix4f t_cam_to_robot0_;
    Eigen::Matrix4f t_robot_to_cam_;

    Object_Recognition object_recognition_;

    std::vector<Object> objects_position_;

    double imu_x0_, imu_y0_, imu_z0_;
    bool imu_calibrated_;

    Eigen::Matrix3f robot_pose_;

    int n_concave_, n_obstacle_;

    bool detectObject(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                      std::string &object_id, pcl::PointXY &object_position_world_frame);

    int image_analysis(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                                          cv::Mat &color_mask, pcl::PointXYZ &position, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                          cv::Mat &floor_mask,
                                          cv::Mat &color_ROI);

    void get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out);
    void backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                            cv::Mat &floor_mask);

    int color_analysis(const cv::Mat &bgr_img,
                             cv::Point2i &mass_center,
                             cv::Mat &out_img);
    void load_calibration(const std::string &path);

    void publish_evidence(const std::string &object_id,
                          const cv::Mat &image);
    void publish_obstacle(bool is_obstacle);
    void publish_object(const std::string &id, const pcl::PointXY position);
    bool detectObstacle(const cv::Mat &floor_mask);

    double estimateDepth(const cv::Mat &depth_img, cv::Point mass_center);

    bool is_new_object(const pcl::PointXY &position);

    bool is_concave(const cv::Mat &depth_img, const cv::Mat &mask_img);

    void publish_markers();
};
}
