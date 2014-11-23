#include <ros/ros.h>
#include <ras_utils/basic_node.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

class Marker_Placement : rob::BasicNode
{
public:
    Marker_Placement();

    void run();

private:
    ros::Publisher marker_pub_;

    tf::TransformBroadcaster br;
    tf::Transform transform;
};

int main(int argc, char* argv[])
{
    // ** Init node
    ros::init(argc, argv, "marker_test");

    // ** Create wall follower object
    Marker_Placement marker;

    // ** Run
    marker.run();
}

Marker_Placement::Marker_Placement()
{
    marker_pub_ = n.advertise<visualization_msgs::MarkerArray>( "my_marker", 100 );
}
void Marker_Placement::run()
{
    ros::Rate r(10);
    while(ros::ok())
    {
        ROS_INFO("PUBLISHING");
        transform.setOrigin( tf::Vector3(0.0, 2.0, 0.0) );
        transform.setRotation( tf::Quaternion(0, 0, 0, 1) );
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/world", "/robot_base"));

        visualization_msgs::MarkerArray markerArray;
        markerArray.markers.resize(2);
        std::vector<int> shapes = {visualization_msgs::Marker::CUBE, visualization_msgs::Marker::TEXT_VIEW_FACING};
        for(int i= 0;i< 2;++i)
        {
            visualization_msgs::Marker &marker = markerArray.markers[i];
            marker.header.frame_id = "/world";
            marker.header.stamp = ros::Time();
            marker.ns = "Marker placement";
            marker.id = i;
            marker.type = shapes[i];
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = 1.0;
            marker.pose.position.y = 1.0;
            marker.pose.position.z = i;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 1;
            marker.scale.y = 1;
            marker.scale.z = 1;
            marker.color.a = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.text = "Red cube";
        }
        marker_pub_.publish( markerArray );
        ros::spinOnce();
        r.sleep();
    }
}
