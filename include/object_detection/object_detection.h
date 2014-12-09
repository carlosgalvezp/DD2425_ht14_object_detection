#include <iostream>
#include <fstream>
#include <ras_utils/ras_utils.h>
#include <ras_utils/basic_node.h>
#include <ras_utils/pcl_utils.h>
#include <ras_utils/ras_names.h>
#include <ras_utils/ras_types.h>
#include <object_recognition/object_recognition.h>
#include <ras_msgs/RAS_Evidence.h>

#include <object_detection/floor_removal.h>
#include <object_detection/color_object_detection.h>

// ROS
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Imu.h"
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <object_detection/hsv_filter.h>
#include <object_detection/hsv_params.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/LU>

#define START_DELAY 25  //  [frames] Wait this delay before we process images, so that the camera can adjust its brightness

#define QUEUE_SIZE 1            // A too large value can cause more delay; it is preferred to drop frames
#define SCALE_FACTOR    0.25    // Subsample in X and Y to speed-up the point cloud building

// ** ROI (Region of Interest)
#define ROI_MIN_U       50
#define ROI_MAX_U       (IMG_COLS)- 80
#define ROI_MIN_V       50
#define ROI_MAX_V       (IMG_ROWS - 50)

// ** Object detection

// Geometric parameters
#define ROBOT_BORDER            0.115                     // [m] Distance from camera to robot front border
#define D_OBJECT_DETECTION_MAX  ROBOT_BORDER + 0.22
#define D_OBJECT_DETECTION_MIN  ROBOT_BORDER + 0.16       // [m] Distance at which we start trying to detect the object
#define NEW_OBJECT_MIN_DISTANCE 0.2                       // [m] Min distance between objects

#define N_MAX_CLASSIFICATIONS 10 // Max number of classifications (avoid memory issues)
namespace object_detection
{
class Object_Detection : rob::BasicNode{

    struct Object
    {
        pcl::PointXY position_;
        std::string id_;

        Object(pcl::PointXY position, const std::string &id)
            :position_(position), id_(id){}
    };
public:
    Object_Detection();

    void RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                         const sensor_msgs::ImageConstPtr &depth_msg);   
    void saveRobotPositions(const string &path);
private:
    message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    boost::shared_ptr<RAS_Types::RGBD_Sync> rgbd_sync_;

    ros::Publisher speaker_pub_, evidence_pub_, marker_pub_, robot_position_pub_;
    ros::Publisher pcl_pub_, object_pcl_pub_;

    int frame_counter_;
    cv::Mat ROI_;

    Eigen::Matrix4f t_cam_to_robot_;
    Eigen::Matrix4f t_robot_to_cam_;
    Eigen::Matrix4f t_robot_to_world_;
    tf::TransformListener tf_listener_;

    Object_Recognition object_recognition_;

    std::vector<Object> objects_position_;
    std::vector<geometry_msgs::Point> robot_positions_;

    HSV_Filter hsv_filter_;

    std::vector<std::string> classifications_;


    int phase_, contest_;

    bool readTransforms();
    void createROI(cv::Mat &img);
    bool detectObject(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                      std::string &object_id, pcl::PointXY &object_position_world_frame);


    void publish_evidence(const std::string &object_id,
                          const cv::Mat &image);
    void publish_object(const std::string &id, const pcl::PointXY &position);
    void add_robot_position();
    void publish_markers();

    double estimateDepth(const cv::Mat &depth_img, cv::Point mass_center);

    bool is_new_object(const pcl::PointXY &position);

};
}
