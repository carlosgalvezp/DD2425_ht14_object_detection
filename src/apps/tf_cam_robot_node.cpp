#include <fstream>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sensor_msgs/Imu.h>

#include <ras_utils/basic_node.h>
#include <ras_utils/ras_utils.h>
#include <ras_utils/ras_names.h>
#include <ras_utils/pcl_utils.h>

#include <Eigen/Core>

#define PUBLISH_RATE 50

class TF_Cam_Robot : rob::BasicNode
{
public:
    TF_Cam_Robot();
    void run();
private:
    void IMUCallback(const sensor_msgs::ImuConstPtr &imu_msg);
    void loadTransform(const std::string &path, Eigen::Matrix4f &tf);

    ros::Subscriber imu_sub_;
    tf::TransformBroadcaster tf_broadcaster_;

    Eigen::Matrix4f t_cam_to_robot0_, t_cam_to_robot_;
    bool imu_calibrated_;

    double imu_x0_, imu_y0_, imu_z0_;
};

int main(int argc, char* argv[])
{
    // ** Init node
    ros::init(argc, argv, "obstacle_detection");

    // ** Create object recognition object
    TF_Cam_Robot o;

    o.run();

    return 0;
}

// =============================================================================
// =============================================================================
TF_Cam_Robot::TF_Cam_Robot()
    : imu_calibrated_(false)
{
    imu_sub_ = n.subscribe(TOPIC_IMU, 10, &TF_Cam_Robot::IMUCallback, this);

    this->loadTransform(RAS_Names::CALIBRATION_PATH, this->t_cam_to_robot0_);
    this->t_cam_to_robot_ = this->t_cam_to_robot0_;
}

void TF_Cam_Robot::run()
{
    ros::Rate r(PUBLISH_RATE);
    while(ros::ok())
    {
        tf::Transform transform;
        PCL_Utils::convertEigen4x4ToTransform(this->t_cam_to_robot_, transform);

        tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), COORD_FRAME_ROBOT,COORD_FRAME_CAMERA_LINK));

        r.sleep();
        ros::spinOnce();
    }
}

void TF_Cam_Robot::IMUCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double x,y,z;
    x = imu_msg->linear_acceleration.x;
    y = imu_msg->linear_acceleration.y;
    z = imu_msg->linear_acceleration.z;

    // ** Calibrate if required
    if (!imu_calibrated_)
    {
        imu_x0_ = x;
        imu_y0_ = y;
        imu_z0_ = z;
        imu_calibrated_ = true;
    }

    // ** Compute angle (rotation around y axis)
//    double theta_y = asin(fabs((x-imu_x0_)/imu_z0_));
    double theta_y = atan(fabs((x-imu_x0_)/z));
    Eigen::Matrix4f t_tilt_;
    t_tilt_ << cos(theta_y),     0,      sin(theta_y), 0,
            0,                1,      0,               0,
            -sin(theta_y),    0,      cos(theta_y),    0,
            0,                0,                0,     1;

    t_cam_to_robot_ = t_tilt_ * t_cam_to_robot0_;
}

void TF_Cam_Robot::loadTransform(const std::string &path, Eigen::Matrix4f &tf)
{
    std::ifstream file(path);

    tf = Eigen::Matrix4f::Identity();
    for(unsigned int i = 0; i < 4; ++i)
    {
        for(unsigned int j = 0; j < 4; ++j)
        {
            file >> tf(i,j);
        }
    }
    std::cout << "Read calibration " << std::endl << tf <<std::endl;
    file.close();
}
