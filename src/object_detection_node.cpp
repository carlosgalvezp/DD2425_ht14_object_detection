#include <object_detection/object_detection.h>

int main(int argc, char* argv[])
{
    // ** Init node
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle n;
    ros::NodeHandle n_private("~");

    // ** Create object recognition object
    object_detection::Object_Detection o(n, n_private);

    // ** Spin
    ros::spin();
    return 0;
}
