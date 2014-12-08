#include <object_detection/object_detection.h>

namespace object_detection
{
Object_Detection::Object_Detection()
    : frame_counter_(0)
{ 
    // ** Publishers
    evidence_pub_ = n.advertise<ras_msgs::RAS_Evidence>(TOPIC_EVIDENCE, 2);
    speaker_pub_  = n.advertise<std_msgs::String>(TOPIC_SPEAKER, 2);
    marker_pub_   = n.advertise<visualization_msgs::MarkerArray>(TOPIC_OBJECT_MARKERS, 2);

    pcl_pub_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/object_detection/cloud", 10);
    object_pcl_pub_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/object_detection/object", 10);

    robot_position_pub_ = n.advertise<geometry_msgs::Point>(TOPIC_ROBOT_OBJECT_POSITION,2);

    // ** Subscribers
    rgb_sub_.subscribe(n, TOPIC_CAMERA_RGB, QUEUE_SIZE);
    depth_sub_.subscribe(n, TOPIC_CAMERA_DEPTH, QUEUE_SIZE);
    rgbd_sync_.reset(new RAS_Types::RGBD_Sync(RAS_Types::RGBD_Sync_Policy(QUEUE_SIZE), rgb_sub_, depth_sub_));
    rgbd_sync_->registerCallback(boost::bind(&Object_Detection::RGBD_Callback, this, _1, _2));

    // ** ROI
    createROI(this->ROI_);

    // ** Param server
    n.getParam(PARAM_PHASE, this->phase_);
    n.getParam(PARAM_CONTEST, this->contest_);

    // ** HSV params
    if(this->contest_ == 1)
        this->hsv_filter_ = HSV_Filter(RAS_Names::HSV_PARAMS_CONTEST);
    else
        this->hsv_filter_ = HSV_Filter(RAS_Names::HSV_PARAMS_LAB);

    // ** Object recognition
    object_recognition_ = Object_Recognition(object_pcl_pub_);
}


void Object_Detection::RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                     const sensor_msgs::ImageConstPtr &depth_msg)
{
    ros::WallTime t1(ros::WallTime::now());
    if(frame_counter_ > START_DELAY && readTransforms())
    {
        // ** Convert ROS messages to OpenCV images and scale
        cv_bridge::CvImageConstPtr rgb_ptr   = cv_bridge::toCvShare(rgb_msg);
        cv_bridge::CvImageConstPtr depth_ptr   = cv_bridge::toCvShare(depth_msg);
        const cv::Mat& rgb_img     = rgb_ptr->image;
        const cv::Mat& depth_img   = depth_ptr->image;

        // ** Detect Object
        std::string object_id;
        pcl::PointXY object_position_world_frame;
        bool foundObject = detectObject(rgb_img, depth_img, object_id, object_position_world_frame);

        // ** Publish evidence if found object
        if(foundObject)
        {
            // ** Update the map
            objects_position_.push_back(Object(object_position_world_frame, object_id));

            // ** Publish evidence, markers on the map and add robot position for global path planning
            publish_evidence(object_id, rgb_img);
            publish_markers();
            add_robot_position();
        }
    }
    else
        frame_counter_++;
    ROS_INFO("[ObjectDetection] %.3f ms", RAS_Utils::time_diff_ms(t1, ros::WallTime::now()));
}

bool Object_Detection::detectObject(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                                    std::string &object_id, pcl::PointXY &object_position_world_frame)
{
    // ** Remove floor and use ROI
    Floor_Removal floor_removal;    cv::Mat floor_mask;
    floor_removal.remove_floor(bgr_img, depth_img, SCALE_FACTOR, this->t_cam_to_robot_, this->t_robot_to_cam_,
                               this->pcl_pub_, floor_mask);
    cv::bitwise_and(floor_mask, this->ROI_, floor_mask);

    Color_Object_Detection color_detection(this->hsv_filter_);
    cv::Mat object_mask; cv::Point mass_center;
    int color = color_detection.detectColoredObject(bgr_img, floor_mask, object_mask, mass_center);

    cv::imshow("Object mask", object_mask);
    cv::waitKey(1);

    // ** Get 3D position of the object
    if(color >=0)
    {
        pcl::PointXYZ object_position_robot_frame, object_position_cam_frame;
        double depth = estimateDepth(depth_img, mass_center);
        PCL_Utils::transform2Dto3D(mass_center, depth, object_position_cam_frame);
        PCL_Utils::transformPoint(object_position_cam_frame, t_cam_to_robot_, object_position_robot_frame);

        std::cout << "Distance to robot: "<<object_position_robot_frame.x
                  <<"; MAX: "<< D_OBJECT_DETECTION_MAX
                  <<"; MIN: "<< D_OBJECT_DETECTION_MIN<<std::endl;


        // ** Call the recognition node if object found
        if ( object_position_robot_frame.x < D_OBJECT_DETECTION_MAX &&
            (object_position_robot_frame.x > D_OBJECT_DETECTION_MIN || classifications_.size() != 0))
        {

            // ** Transform into world frame and see if we have seen this before
            pcl::PointXYZ tmp_object_position_world_frame;
            PCL_Utils::transformPoint(object_position_robot_frame, t_robot_to_world_, tmp_object_position_world_frame);
            object_position_world_frame.x = tmp_object_position_world_frame.x;
            object_position_world_frame.y = tmp_object_position_world_frame.y;

            // ** Call recognition if it's a new object
            if(is_new_object(object_position_world_frame))
            {
                std::string tmp_classification;

                object_recognition_.classify(bgr_img, depth_img, object_mask, t_cam_to_robot_, tmp_classification);
                classifications_.push_back(tmp_classification);

                // ** Get the msot likely result when we are now too close to the object
                if(object_position_robot_frame.x < D_OBJECT_DETECTION_MIN || classifications_.size() > N_MAX_CLASSIFICATIONS)
                {
                    object_id = RAS_Utils::get_most_repeated<std::string>(classifications_);
                    classifications_.clear();
                    return true;
                }
            }
        }
    }
    else
    {
        classifications_.clear();
    }
    return false;
}

// ** ==================== HELP FUNCTIONS ==============================
bool Object_Detection::readTransforms()
{
    tf::Transform tf_cam_to_robot, tf_robot_to_world;
    bool t1 = PCL_Utils::readTransform(COORD_FRAME_ROBOT, COORD_FRAME_CAMERA_RGB_OPTICAL, this->tf_listener_, tf_cam_to_robot);
    bool t2 = PCL_Utils::readTransform(COORD_FRAME_WORLD, COORD_FRAME_ROBOT, this->tf_listener_, tf_robot_to_world);

    if(t1 && t2)
    {
        PCL_Utils::convertTransformToEigen4x4(tf_cam_to_robot, this->t_cam_to_robot_);
        PCL_Utils::convertTransformToEigen4x4(tf_robot_to_world, this->t_robot_to_world_);
        this->t_robot_to_cam_ = this->t_cam_to_robot_.inverse();
    }
    else
        return false;
}

void Object_Detection::publish_evidence(const std::string &object_id, const cv::Mat &image)
{
    ros::Time stamp = ros::Time::now();
    // ** Convert cv::Mat to sensor_msgs/Image
    cv_bridge::CvImage rgb_img;
    rgb_img.header.stamp = stamp;
    rgb_img.encoding = sensor_msgs::image_encodings::RGB8;
    rgb_img.image = image;

    // ** Create messages
    ras_msgs::RAS_Evidence msg;
    msg.stamp = stamp;
    msg.group_number = GROUP_NUMBER;
    msg.image_evidence = *rgb_img.toImageMsg();
    msg.object_id = object_id;

    std_msgs::String speaker_msg;
    if(object_id != OBJECT_NAME_PATRIC && object_id != OBJECT_NAME_UNKNOWN)
        speaker_msg.data = "I see a " + object_id;
    else
        speaker_msg.data = "I see " + object_id;

    evidence_pub_.publish(msg);
    speaker_pub_.publish(speaker_msg);
}

void Object_Detection::publish_object(const std::string &id, const pcl::PointXY &position)
{
    ROS_ERROR("TO DO");
}

void Object_Detection::add_robot_position()
{
    geometry_msgs::Point position;
    position.x = t_robot_to_world_(0,3);
    position.y = t_robot_to_world_(1,3);

    robot_positions_.push_back(position);
}

bool Object_Detection::is_new_object(const pcl::PointXY &position)
{
    for(std::size_t i = 0; i < objects_position_.size(); ++i)
    {
        double d =PCL_Utils::euclideanDistance(position, objects_position_[i].position_);
        if(d < NEW_OBJECT_MIN_DISTANCE)
        {
            return false;
        }
    }
    return true;
}

void Object_Detection::saveRobotPositions(const std::string &path)
{
    std::ofstream file(path);

    file << robot_positions_.size()<<std::endl;
    for(std::size_t i = 0; i < robot_positions_.size(); ++i)
    {
        const geometry_msgs::Point &obj = robot_positions_[i];
        file << obj.x << " "<<obj.y<<std::endl;
    }
    file.close();
}

double Object_Detection::estimateDepth(const cv::Mat &depth_img, cv::Point mass_center)
{
    // Start at mass center and go vertically until not NaN is found. This
    // is especially suited for "invisible" objects, since it will compute
    // the depth at the contact point between the floor and the object
    double depth=0;
    for(unsigned int i = mass_center.y; i < IMG_ROWS; ++i)
    {
        double d = depth_img.at<float>(i,mass_center.x);
        if( d!= 0 && !std::isnan(d))
        {
            depth = d;
            break;
        }
    }
    return depth;
}

void Object_Detection::publish_markers()
{
    visualization_msgs::MarkerArray msg;
    msg.markers.resize(objects_position_.size()*2); // Display object and ID text

    for(std::size_t i=0; i < msg.markers.size(); ++i)
    {        
        visualization_msgs::Marker & marker_obj = msg.markers[i];
        Object & obj = objects_position_[i%objects_position_.size()];

        double z = i < objects_position_.size() ? 0 : 0.1;
        marker_obj.header.frame_id = COORD_FRAME_WORLD;
        marker_obj.header.stamp = ros::Time();
        marker_obj.ns = RVIZ_MARKER_NS_OBJECT;
        marker_obj.id = i;
        marker_obj.action = visualization_msgs::Marker::ADD;
        marker_obj.pose.position.x = obj.position_.x; //m
        marker_obj.pose.position.y = obj.position_.y; //m
        marker_obj.pose.position.z = z + 0.025; //m

        marker_obj.pose.orientation.x = 0.0;
        marker_obj.pose.orientation.y = 0.0;
        marker_obj.pose.orientation.z = 0.0;
        marker_obj.pose.orientation.w = 1.0;
        marker_obj.scale.x = 0.05;
        marker_obj.scale.y = 0.05;
        marker_obj.scale.z = 0.05;
        marker_obj.color.a = 1.0;
        marker_obj.color.r = i >= objects_position_.size();
        marker_obj.color.g = i < objects_position_.size();
        marker_obj.color.b = 0.0;
        if (i < objects_position_.size())
            marker_obj.type = visualization_msgs::Marker::CUBE;
        else
            marker_obj.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker_obj.text = obj.id_;

    }
    marker_pub_.publish( msg);
}

void Object_Detection::createROI(cv::Mat &img)
{
    img = cv::Mat::zeros(IMG_ROWS, IMG_COLS, CV_8UC1);
    for(unsigned int v = 0; v < img.rows; ++v)
    {
        for(unsigned int u = 0; u < img.cols; ++u)
        {
            if(u >= ROI_MIN_U && u <= ROI_MAX_U && v >= ROI_MIN_V && v <= ROI_MAX_V)
                img.at<uint8_t>(v,u) = 255;
        }
    }
}
}  // namespace
