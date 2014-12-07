#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>

#include <ras_arduino_msgs/ADConverter.h>

#include <ras_utils/basic_node.h>
#include <ras_utils/ras_utils.h>
#include <ras_utils/ras_names.h>
#include <ras_utils/pcl_utils.h>
#include <ras_utils/ras_sensor_utils.h>

#include <Eigen/Core>

#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>

#define START_DELAY 25  //  [frames] Wait this delay before we process images, so that the camera can adjust its brightness
#define SCALE_FACTOR 0.5

#define ROBOT_WIDTH 0.23 // [m]

#define MAX_DEPTH   0.5 // [m]
#define RESOLUTION  0.005 // [m]

class Obstacle_Detection : rob::BasicNode
{
public:
    Obstacle_Detection();

private:
    void adcCallback(const ras_arduino_msgs::ADConverterConstPtr &msg);
    void depthCallback(const sensor_msgs::Image::ConstPtr &img);
    void extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in, std::vector<double> &distances);
    double getLineDepth(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &line_cloud);

    int frame_counter_;
    ros::Publisher obstacle_pub_;
    ros::Subscriber depth_sub_, adc_sub_;
    tf::TransformListener tf_listener_;
    double front_sensor_distance_; // [m]

};

int main(int argc, char* argv[])
{
    // ** Init node
    ros::init(argc, argv, "obstacle_detection");

    // ** Create object recognition object
    Obstacle_Detection o;

    // ** Spin
    ros::spin();

    return 0;
}

// =============================================================================
// =============================================================================
Obstacle_Detection::Obstacle_Detection()
    : frame_counter_(0)
{
    depth_sub_ = n.subscribe(TOPIC_CAMERA_DEPTH, 2, &Obstacle_Detection::depthCallback, this);
    adc_sub_   = n.subscribe(TOPIC_ARDUINO_ADC, 2, &Obstacle_Detection::adcCallback, this);
}

void Obstacle_Detection::depthCallback(const sensor_msgs::Image::ConstPtr &img)
{
    tf::Transform tf_cam_to_robot;
    if(frame_counter_ > START_DELAY && front_sensor_distance_ > MAX_DEPTH
       && PCL_Utils::readTransform(COORD_FRAME_CAMERA_LINK, COORD_FRAME_ROBOT, tf_listener_, tf_cam_to_robot))
    {
        // ** Convert ROS messages to OpenCV images and scale
        cv_bridge::CvImageConstPtr depth_ptr   = cv_bridge::toCvShare(img);
        const cv::Mat& depth_img   = depth_ptr->image;

        // ** Build point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        PCL_Utils::buildPointCloud(depth_img, cloud, SCALE_FACTOR);

        // ** Get transformation to robot coordinate frame
        Eigen::Matrix4f eigen_cam_to_robot;
        PCL_Utils::convertTransformToEigen4x4(tf_cam_to_robot, eigen_cam_to_robot);
        // ** Transform point cloud
        pcl::transformPointCloud(*cloud, *cloud, eigen_cam_to_robot);

        // ** Extract obstacles
        std::vector<double> distances;
        extractObstacles(cloud, distances);

        // ** Publish
        ROS_ERROR("TO DO PUBLISH");
    }
}

void Obstacle_Detection::extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
                                          std::vector<double> &distances)
{
    // ** Pass-through filter
    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.03, 0.1);
    pass.filter (*cloud_filtered);

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-ROBOT_WIDTH/2.0, ROBOT_WIDTH/2.0);
    pass.filter (*cloud_filtered);


    // ** Analize each line
    for(double i = -ROBOT_WIDTH/2.0; i < ROBOT_WIDTH/2.0; i+=RESOLUTION)
    {
        // ** Create line
        pcl::PointCloud<pcl::PointXYZ>::Ptr line(new pcl::PointCloud<pcl::PointXYZ>);
        pass.setInputCloud (cloud_filtered);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (i, i + RESOLUTION);
        pass.filter (*line);

        double d = getLineDepth(line);
        distances.push_back(d);
    }
}

double Obstacle_Detection::getLineDepth(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &line_cloud)
{
    if(line_cloud->size() != 0)
    {
        double min_d = MAX_DEPTH;
        for(std::size_t i = 0; i < line_cloud->size(); ++i)
        {
            const pcl::PointXYZ &p = line_cloud->points[i];
            double depth = p.x;
            if(depth < min_d)
            {
                min_d = depth;
            }

        }
        return min_d;
    }
    return MAX_DEPTH;
}

void Obstacle_Detection::adcCallback(const ras_arduino_msgs::ADConverterConstPtr &msg)
{
    int adc = msg->ch8;
    this->front_sensor_distance_ = 0.01*RAS_Utils::sensors::longSensorToDistanceInCM(adc) + ROBOT_WIDTH/2.0;
}
