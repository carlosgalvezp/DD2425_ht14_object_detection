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

#include <ras_srv_msgs/LaserLine.h>
#include <ras_srv_msgs/LaserScanner.h>

#include <Eigen/Core>

#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl_ros/point_cloud.h>

#define START_DELAY     25  //  [frames] Wait this delay before we process images, so that the camera can adjust its brightness
#define SCALE_FACTOR    0.5

#define ROBOT_WIDTH     0.23 // [m]

#define MIN_DEPTH       0.2                // [m]
#define MAX_DEPTH       0.4                // [m]
#define LASER_WIDTH     ROBOT_WIDTH        // [m]
#define LASER_HEIGHT    0.05               // [m]
#define RESOLUTION      0.005              // [m]


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
    void extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
                          const tf::Transform &tf_robot_to_world,
                          std::vector<Line_Segment> &distances);
    double getLineDepth(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &line_cloud);

    void publishLines(const std::vector<Line_Segment> &lines);

    int frame_counter_;
    ros::Publisher obstacle_pub_, pcl_pub_, map_pub_;
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
    map_pub_ = n.advertise<ras_srv_msgs::LaserScanner>(TOPIC_OBSTACLE_LASER_MAP,2);
}

void Obstacle_Detection::depthCallback(const sensor_msgs::Image::ConstPtr &img)
{
    ros::WallTime t1(ros::WallTime::now());
    tf::Transform tf_cam_to_robot, tf_robot_to_world;

    if(frame_counter_ > START_DELAY
       && PCL_Utils::readTransform(COORD_FRAME_ROBOT, COORD_FRAME_CAMERA_RGB_OPTICAL, tf_listener_, tf_cam_to_robot)
       && PCL_Utils::readTransform(COORD_FRAME_WORLD, COORD_FRAME_ROBOT, tf_listener_, tf_robot_to_world))
    {
        // ** Convert ROS messages to OpenCV images and scale
        cv_bridge::CvImageConstPtr depth_ptr   = cv_bridge::toCvShare(img);
        const cv::Mat& depth_img   = depth_ptr->image;

        // ** Build point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        PCL_Utils::buildPointCloud(depth_img, cloud, SCALE_FACTOR);

        // ** Transform point cloud
        Eigen::Matrix4f eigen_cam_to_robot;
        PCL_Utils::convertTransformToEigen4x4(tf_cam_to_robot, eigen_cam_to_robot);
        pcl::transformPointCloud(*cloud, *cloud, eigen_cam_to_robot);
        cloud->header.frame_id = COORD_FRAME_WORLD;
        pcl_pub_.publish(cloud);

        // ** Extract obstacles
        std::vector<Line_Segment> distances;
        extractObstacles(cloud, tf_robot_to_world, distances);

        // ** Publish
        publishLines(distances);
    }
    else
    {
        ++frame_counter_;
    }
//    ROS_INFO("[LaserScanner] %.3f ms", RAS_Utils::time_diff_ms(t1, ros::WallTime::now()));
}

void Obstacle_Detection::extractObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud_in,
                                          const tf::Transform &tf_robot_to_world,
                                          std::vector<Line_Segment> &distances)
{
    // ** Pass-through filter
    pcl::PassThrough<pcl::PointXYZ> pass;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (LASER_HEIGHT, LASER_HEIGHT + 0.01);
    pass.filter (*cloud_filtered);


    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-LASER_WIDTH/2.0, LASER_WIDTH/2.0);
    pass.filter (*cloud_filtered);


    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (MIN_DEPTH, MAX_DEPTH);
    pass.filter (*cloud_filtered);

    Eigen::Matrix4f tf_eigen;
    PCL_Utils::convertTransformToEigen4x4(tf_robot_to_world, tf_eigen);

    // ** Analize each line
    for(double i = -LASER_WIDTH/2.0; i < LASER_WIDTH/2.0; i+=RESOLUTION)
    {
        // ** Create line
        pcl::PointCloud<pcl::PointXYZ>::Ptr line(new pcl::PointCloud<pcl::PointXYZ>);
        pass.setInputCloud (cloud_filtered);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (i, i + RESOLUTION);
        pass.filter (*line);

        double d = getLineDepth(line);
        geometry_msgs::Point from, to;
        from.x = MIN_DEPTH; from.y = i; from.z = LASER_HEIGHT;
        to.x   = d;         to.y   = i; to.z   = LASER_HEIGHT;

        // ** Transform into world frame
        pcl::PointXYZ from3d(from.x, from.y, from.z);
        pcl::PointXYZ to3d(to.x, to.y, to.z);
        PCL_Utils::transformPoint(from3d, tf_eigen, from3d);
        PCL_Utils::transformPoint(to3d, tf_eigen, to3d);

        from.x = from3d.x;        from.y = from3d.y;
        to.x = to3d.x;            to.y = to3d.y;
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
    visualization_msgs::MarkerArray msg;
    ras_srv_msgs::LaserScanner map_msg;
    msg.markers.resize(lines.size());
    map_msg.scan.resize(lines.size());

    for(std::size_t i = 0; i < lines.size(); ++i)
    {
        visualization_msgs::Marker &m = msg.markers[i];
        const Line_Segment &l = lines[i];

        m.header.frame_id = COORD_FRAME_WORLD;
        m.header.stamp = ros::Time();
        m.ns = "laser_scanner";
        m.id = i;
        m.action = visualization_msgs::Marker::ADD;

        m.scale.x = RESOLUTION;

        // ** Get point and transform into world coordinate
        m.points.clear();
        m.points.push_back(l.from_);
        m.points.push_back(l.to_);

        m.color.a = 1.0;
        m.color.r = l.is_wall_ ? 1.0 : 0.0;
        m.color.g = 1.0 - m.color.r;
        m.color.b = 0.0;
        m.type = visualization_msgs::Marker::LINE_STRIP;

        // ** Fill in data for map
        ras_srv_msgs::LaserLine &laser_line = map_msg.scan[i];
        laser_line.from_ = l.from_;
        laser_line.to_ = l.to_;
        laser_line.is_wall = l.is_wall_;
    }
    obstacle_pub_.publish(msg);
    map_pub_.publish(map_msg);
}

void Obstacle_Detection::adcCallback(const ras_arduino_msgs::ADConverterConstPtr &msg)
{
    int adc = msg->ch8;
    this->front_sensor_distance_ = 0.01*RAS_Utils::sensors::longSensorToDistanceInCM(adc) + ROBOT_WIDTH/2.0;
}


