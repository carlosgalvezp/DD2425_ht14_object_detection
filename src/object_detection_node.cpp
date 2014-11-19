#include <object_detection/object_detection.h>

int main(int argc, char* argv[])
{
    // ** Init node
    ros::init(argc, argv, "object_detection");

    // ** Create object recognition object
    object_detection::Object_Detection o;

    // ** Spin
    ros::spin();
    return 0;
}
