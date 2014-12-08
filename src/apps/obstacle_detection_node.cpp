#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>

#include <visualization_msgs/MarkerArray.h>
#include <ras_arduino_msgs/ADConverter.h>

#include <ras_utils/basic_node.h>
#include <ras_utils/ras_utils.h>
#include <ras_utils/ras_names.h>
#include <ras_utils/pcl_utils.h>
#include <ras_utils/ras_sensor_utils.h>

#include <Eigen/Core>

#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>

#define START_DELAY 25  //  [frames] Wait this delay before we process images, so that the camera can adjust its brightness
#define SCALE_FACTOR 0.5

#define ROBOT_WIDTH 0.23 // [m]

#define MIN_DEPTH   0.2 // [m]
#define MAX_DEPTH   0.5 // [m]
#define RESOLUTION  0.01 // [m]

#define MAP_WIDTH  1000
#define MAP_HEIGHT 1000
#define MAP_RESOLUTION 0.01

class Obstacle_Detection : rob::BasicNode
{
struct Line_Segment
{
    geometry_msgs::Point from_, to_;
    bool is_wall_;

    Line_Segment(const geometry_msgs::Point &from,
                 const geometry_msgs::Point &to,
                 bool is_wall)
        : from_(from), to_(to), is_wall_(is_wall){}
};

public:
    Obstacle_Detection();

private:
    void adcCallback(const ras_arduino_msgs::ADConverterConstPtr &msg);
    void depthCallback(const sensor_msgs::Image::ConstPtr &img);
    void extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in, std::vector<Line_Segment> &distances);
    double getLineDepth(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &line_cloud);

    void publishLines(const std::vector<Line_Segment> &lines);

    int frame_counter_;
    ros::Publisher obstacle_pub_, pcl_pub_;
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

    pcl_pub_ = n.advertise<pcl::PointCloud<pcl::PointXYZ> >("/obstacle_detection/cloud", 2);
    obstacle_pub_ = n.advertise<visualization_msgs::MarkerArray>("/laser_scanner",2);
}

void Obstacle_Detection::depthCallback(const sensor_msgs::Image::ConstPtr &img)
{
    tf::Transform tf_cam_to_robot;
    if(frame_counter_ > START_DELAY && front_sensor_distance_ > MAX_DEPTH
       && PCL_Utils::readTransform(COORD_FRAME_ROBOT, COORD_FRAME_CAMERA_RGB_OPTICAL, tf_listener_, tf_cam_to_robot))
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
        std::vector<Line_Segment> distances;
        extractObstacles(cloud, distances);

        // ** Publish
        publishLines(distances);
    }
}

void Obstacle_Detection::extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
                                          std::vector<Line_Segment> &distances)
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

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (MIN_DEPTH, MAX_DEPTH);
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
        geometry_msgs::Point from, to;
        from.x = MIN_DEPTH; from.y = i; from.z = 0;
        to.x   = d;         to.y   = i; to.z   = 0;
        Line_Segment l(from, to, d < MAX_DEPTH);
        distances.push_back(l);
    }
}

double Obstacle_Detection::getLineDepth(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &line_cloud)
{
    if(line_cloud->size() != 0) // There's points in the line
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

void Obstacle_Detection::publishLines(const std::vector<Line_Segment> &lines)
{
    // ** Get robot transform
    tf::Transform tf_robot_to_world;
    if(PCL_Utils::readTransform(COORD_FRAME_WORLD, COORD_FRAME_ROBOT, this->tf_listener_, tf_robot_to_world))
    {
        double x_robot = tf_robot_to_world.getOrigin()[0];
        double y_robot = tf_robot_to_world.getOrigin()[1];

        visualization_msgs::MarkerArray msg;
        msg.markers.resize(lines.size());

        for(std::size_t i = 0; i < lines.size(); ++i)
        {
            visualization_msgs::Marker &m = msg.markers[i];
            const Line_Segment &l = lines[i];

            m.header.frame_id = COORD_FRAME_WORLD;
            m.header.stamp = ros::Time();
            m.ns = "laser_scanner";
            m.id = i;
            m.action = visualization_msgs::Marker::ADD;

            m.scale.x = 0.01;
            m.scale.y = 0.01;
            m.scale.z = 0.01;

            // ** Get point and transform into world coordinate
            geometry_msgs::Point from = l.from_;
            from.x += x_robot;
            from.y += y_robot;

            geometry_msgs::Point to = l.to_;
            to.x += x_robot;
            to.y += y_robot;

            m.points.push_back(from);
            m.points.push_back(to);

            m.color.a = 1.0;
            m.color.r = l.is_wall_ ? 1.0 : 0.0;
            m.color.g = 1.0 - m.color.r;
            m.color.b = 0.0;
            m.type = visualization_msgs::Marker::LINE_STRIP;
        }
    }

}

void Obstacle_Detection::adcCallback(const ras_arduino_msgs::ADConverterConstPtr &msg)
{
    int adc = msg->ch8;
    this->front_sensor_distance_ = 0.01*RAS_Utils::sensors::longSensorToDistanceInCM(adc) + ROBOT_WIDTH/2.0;
}


