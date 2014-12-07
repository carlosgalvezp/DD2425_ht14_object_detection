#include <object_detection/floor_removal.h>

Floor_Removal::Floor_Removal()
{
}

void Floor_Removal::getFloorMask(const cv::Mat &bgr_img,
                                 const cv::Mat &depth_img,
                                 double scale_factor,
                                 const Eigen::Matrix4f &t_cam_to_robot,
                                 const Eigen::Matrix4f &t_robot_to_cam,
                                 const ros::Publisher &cloud_pub,
                                 cv::Mat &floor_mask)
{
    // ** Create point cloud and transform into robot coordinates
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    PCL_Utils::buildPointCloud(bgr_img, depth_img, cloud, scale_factor);
    pcl::transformPointCloud(*cloud, *cloud, t_cam_to_robot);
    cloud->header.frame_id = COORD_FRAME_ROBOT;
    cloud_pub.publish(cloud);

    // ** Remove floor
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    get_floor_plane(cloud, floor_cloud);

    // ** Back-project to get binary mask
    backproject_floor(floor_cloud, t_robot_to_cam, floor_mask);
}

void Floor_Removal::get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr   &cloud_out)
{
    // Remove floor and most of the walls
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB>);

    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-10.0, 0.015);
    pass.filter (*cloud_1);

    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.05, 10.0);
    pass.filter (*cloud_2);

    *cloud_out = *cloud_1 + *cloud_2;
}

void Floor_Removal::backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                                      const Eigen::Matrix4f &t_robot_to_cam,
                                            cv::Mat &floor_mask)
{
    floor_mask = 255 * cv::Mat::ones(IMG_ROWS, IMG_COLS, CV_8UC1);
    // ** Transform into camera coordinate frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*floor_cloud, *floor_cloud2, t_robot_to_cam);

    for(std::size_t i = 0; i < floor_cloud2->size(); ++i)
    {
        const pcl::PointXYZRGB& pt = floor_cloud2->points[i];
        if(pt.z != 0)
        {
            int u = (pt.x * FX / (pt.z) + CX);
            int v = (pt.y * FY / (pt.z) + CY);
            floor_mask.at<uint8_t>(v,u) = 0;
        }
    }

    // ** Opening (dilate + erosion)
    int sizeD = 4; // This depends on scaling factor
    int sizeE = 7; // This depends on scaling factor
    cv::Mat elementD = getStructuringElement(cv::MORPH_RECT,cv::Size(2*sizeD+1, 2*sizeD+1),cv::Point( sizeD, sizeD) );
    cv::Mat elementE = getStructuringElement(cv::MORPH_RECT,cv::Size(2*sizeE+1, 2*sizeE+1),cv::Point( sizeE, sizeE) );
    cv::erode(floor_mask, floor_mask, elementE);
    cv::dilate(floor_mask, floor_mask, elementD);
}
