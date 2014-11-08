#include <nodelet/nodelet.h>
#include <object_detection/object_detection.h>

namespace object_detection
{
class Object_Detection_Nodelet : public nodelet::Nodelet
{
public:
    Object_Detection_Nodelet(){}
   ~Object_Detection_Nodelet(){}
    virtual void onInit();
    boost::shared_ptr<Object_Detection> inst_;
};


}

