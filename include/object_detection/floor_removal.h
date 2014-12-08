#ifndef FLOOR_REMOVAL_H
#define FLOOR_REMOVAL_H

#include <ras_utils/pcl_utils.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/point_cloud.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>

#define ROBOT_WIDTH             0.23                      // [m] Robot width

class Floor_Removal
{
public:
    Floor_Removal();

    void remove_floor(const cv::Mat& bgr_img,
                      const cv::Mat &depth_img, double scale_factor,
                      const Eigen::Matrix4f &t_cam_to_robot , const Eigen::Matrix4f &t_robot_to_cam, const ros::Publisher &cloud_pub, cv::Mat &floor_mask);

    void get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr   &cloud_out);

    void backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                           const Eigen::Matrix4f &t_robot_to_cam,
                                 cv::Mat &floor_mask);
};

#endif // FLOOR_REMOVAL_H
