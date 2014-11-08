#include <nodelet/nodelet.h>
#include <object_detection/object_detection_nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace object_detection
{
    void Object_Detection_Nodelet::onInit()
    {
        NODELET_DEBUG("Initializing nodelet");
        inst_.reset(new Object_Detection(getNodeHandle(), getPrivateNodeHandle()));
    }
}

PLUGINLIB_DECLARE_CLASS(object_detection,
                        Object_Detection_Nodelet,
                        object_detection::Object_Detection_Nodelet,
                        nodelet::Nodelet)
