#include <object_detection/object_detection.h>

namespace object_detection
{
Object_Detection::Object_Detection()
    : frame_counter_(0), imu_calibrated_(false), n_concave_(0), n_obstacle_(0)
{
    // ** Load camera extrinsic calibration
    load_calibration(RAS_Names::CALIBRATION_PATH);

    // ** Publishers
    evidence_pub_ = n.advertise<ras_msgs::RAS_Evidence>(TOPIC_EVIDENCE, 10);
    speaker_pub_ = n.advertise<std_msgs::String>(TOPIC_SPEAKER, 10);
    obstacle_pub_ = n.advertise<std_msgs::Bool>(TOPIC_OBSTACLE, 10);
    marker_pub_ = n.advertise<visualization_msgs::MarkerArray>(TOPIC_MARKERS, 10);

    pcl_pub_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/object_detection/cloud", 2);
    object_pcl_pub_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/object_detection/object", 2);

    // ** Subscribers
    imu_sub_ = n.subscribe(TOPIC_IMU, QUEUE_SIZE,  &Object_Detection::IMUCallback, this);
    odometry_sub_ = n.subscribe(TOPIC_ODOMETRY, QUEUE_SIZE,  &Object_Detection::OdometryCallback, this);
    rgb_sub_.subscribe(n, TOPIC_CAMERA_RGB, QUEUE_SIZE);
    depth_sub_.subscribe(n, TOPIC_CAMERA_DEPTH, QUEUE_SIZE);
    rgbd_sync_.reset(new RGBD_Sync(RGBD_Sync_Policy(QUEUE_SIZE), rgb_sub_, depth_sub_));
    rgbd_sync_->registerCallback(boost::bind(&Object_Detection::RGBD_Callback, this, _1, _2));

    // ** Dynamic reconfigure
    DR_f_ = boost::bind(&Object_Detection::DR_Callback, this, _1, _2);
    DR_server_.setCallback(DR_f_);

    // ** Create ROI
    ROI_ = cv::Mat::zeros(IMG_ROWS, IMG_COLS, CV_8UC1);
    for(unsigned int v = 0; v < ROI_.rows; ++v)
    {
        for(unsigned int u = 0; u < ROI_.cols; ++u)
        {
            if(u >= ROI_MIN_U && u <= ROI_MAX_U && v >= ROI_MIN_V && v <= ROI_MAX_V)
                ROI_.at<uint8_t>(v,u) = 255;
        }
    }

    // ** Object recognition
    object_recognition_ = Object_Recognition(object_pcl_pub_, t_cam_to_robot_);
}


void Object_Detection::RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                     const sensor_msgs::ImageConstPtr &depth_msg)
{
    ros::WallTime t_begin = ros::WallTime::now();
    if(frame_counter_ > 50)
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

            // ** Publish evidence, object (to the map) and obstacle detection
            publish_evidence(object_id, rgb_img);
            publish_object(object_id, object_position_world_frame);
            publish_markers();
        }

    }
    else
        frame_counter_++;

    ros::WallTime t_end = ros::WallTime::now();
    ROS_INFO("[Object Detection] %.3f ms", RAS_Utils::time_diff_ms(t_begin, t_end));
}

bool Object_Detection::detectObject(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                                    std::string &object_id, pcl::PointXY &object_position_world_frame)
{
    bool result = false;
    pcl::PointCloud<pcl::PointXYZRGB>:: Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZ object_position_robot_frame;
    cv::Mat color_mask, floor_mask, rgb_cropped;

    ros::WallTime t_img(ros::WallTime::now());
    int color = image_analysis(bgr_img, depth_img, color_mask, object_position_robot_frame, cloud, floor_mask, rgb_cropped);
    ROS_INFO("Time image anaysis: %.3f ms", RAS_Utils::time_diff_ms(t_img, ros::WallTime::now()));

    cv::imshow("Floor mask", floor_mask);
    cv::imshow("Color mask", color_mask);
    std::cout << "Position: "<<object_position_robot_frame.x << " [Detection: "<<D_OBJECT_DETECTION<< std::endl;
    cv::waitKey(1);

    // ** Call the recognition node if object found
    if (color >= 0)// && position_robot_frame.x < D_OBJECT_DETECTION)
    {
        // ** Transform into world frame and see if we have seen this before
        pcl::PointXY position_robot_coord_2d;
        position_robot_coord_2d.x = object_position_robot_frame.x;
        position_robot_coord_2d.y = object_position_robot_frame.y;
        PCL_Utils::transformPoint(position_robot_coord_2d, robot_pose_, object_position_world_frame);

//            if(is_new_object(position_world_frame))
//            {
            // ** Recognition
            ros::WallTime t_rec(ros::WallTime::now());
//            result = object_recognition_.classify(bgr_img, depth_img, color_mask, object_id);
            ROS_INFO("Time recognition: %.3f ms", RAS_Utils::time_diff_ms(t_rec, ros::WallTime::now()));
    }
    // ** Leverage the work to detect obstacles
    else
    {
        if(detectObstacle(floor_mask))
        {
            ROS_ERROR("OBSTACLE DETECTED");
            n_obstacle_++;
        }
        else
        {
            n_obstacle_ = 0;
        }
        publish_obstacle(n_obstacle_ > 10);
    }
    return result;
}

int Object_Detection::image_analysis(const cv::Mat &bgr_img, const cv::Mat &depth_img,
                                      cv::Mat &color_mask, pcl::PointXYZ &position, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                                      cv::Mat &floor_mask,
                                      cv::Mat &rgb_cropped)
{
    // ** Create point cloud and transform into robot coordinates
    PCL_Utils::buildPointCloud(bgr_img, depth_img, cloud, SCALE_FACTOR);
    pcl::transformPointCloud(*cloud, *cloud, t_cam_to_robot_);
    pcl_pub_.publish(*cloud);

    // ** Remove floor
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    get_floor_plane(cloud, floor_cloud);

    // ** Back-project to get binary mask
    floor_mask = 255*cv::Mat::ones(bgr_img.rows, bgr_img.cols, CV_8UC1);
    backproject_floor(floor_cloud, floor_mask);
    cv::bitwise_and(floor_mask, ROI_, floor_mask); //Combine with ROI

    // ** Apply mask to remove floor in 2D
    cv::Mat rgb_filtered;
    bgr_img.copyTo(rgb_filtered, floor_mask);
    cv::imshow("Image no floor", rgb_filtered);
    cv::waitKey(1);

    // ** Color analysis
    cv::Point2i mass_center;

    ros::WallTime t1(ros::WallTime::now());
    int color = color_analysis(rgb_filtered, mass_center, color_mask);
    ROS_INFO("Time color analysis: %.3f ms", RAS_Utils::time_diff_ms(t1, ros::WallTime::now()));

    // ** Combine color mask and floor mask
    cv::bitwise_and(color_mask, floor_mask, color_mask);

    // ** Get 3D position of the object, if any
    if(color >=0)
    {
        double depth = estimateDepth(depth_img, mass_center);
        PCL_Utils::transform2Dto3D(mass_center, depth, position);
        PCL_Utils::transformPoint(position, t_cam_to_robot_, position);

        // ** Crop image for further processing (another function please!)
        int rect_x = max(0, mass_center.x - CROP_SIZE/2);
        int rect_y = max(0,mass_center.y - CROP_SIZE/2);

        rect_x = min(rect_x, bgr_img.cols -1 - CROP_SIZE);
        rect_y = min(rect_y, bgr_img.rows -1 - CROP_SIZE);

        cv::Rect bbox = cv::Rect(rect_x, rect_y, CROP_SIZE,CROP_SIZE);
        cv::Mat tmp = bgr_img(bbox);
        tmp.copyTo(rgb_cropped);
    }
    return color;
}

void Object_Detection::get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB>);

    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-10.0, 0.015);
    pass.filter (*cloud_1);

//    pass.setInputCloud (cloud_in);
//    pass.setFilterFieldName ("z");
//    pass.setFilterLimits (0.05, 10.0);
//    pass.filter (*cloud_2);

    *cloud_out = *cloud_1;// + *cloud_2;
}

void Object_Detection::backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                                                                                    cv::Mat &floor_mask)
{
    // ** Transform into camera coordinate frame
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*floor_cloud, *floor_cloud2, t_robot_to_cam_);

    for(std::size_t i = 0; i < floor_cloud2->size(); ++i)
    {
        const pcl::PointXYZRGB& pt = floor_cloud2->points[i];
        if(pt.z != 0)
        {
            int u = (pt.x * FX / (pt.z) + CX);
            int v = (pt.y * FY / (pt.z) + CY);
            floor_mask.at<uint8_t>(v,u) = 0;
        }
    }

    // ** Opening (dilate + erosion)
    int sizeD = 4; // This depends on scaling factor
    int sizeE = 7; // This depends on scaling factor
    cv::Mat elementD = getStructuringElement(cv::MORPH_RECT,cv::Size(2*sizeD+1, 2*sizeD+1),cv::Point( sizeD, sizeD) );
    cv::Mat elementE = getStructuringElement(cv::MORPH_RECT,cv::Size(2*sizeE+1, 2*sizeE+1),cv::Point( sizeE, sizeE) );
    cv::erode(floor_mask, floor_mask, elementE);
    cv::dilate(floor_mask, floor_mask, elementD);
}

int Object_Detection::color_analysis(const cv::Mat &bgr_img,
                                           cv::Point2i &mass_center,
                                           cv::Mat &out_img)
{
    out_img = cv::Mat::zeros(bgr_img.rows, bgr_img.cols, CV_8UC1);

    // ** Parameters
    int erode_times=1;
    int dilate_times=1;
    int number_of_colors = hsv_params_.H_min.size();

    // ** Convert to HSV
    cv::Mat img_hsv;
    cv::cvtColor(bgr_img,img_hsv,CV_BGR2HSV);

    // ** Create color masks
    std::vector<cv::Mat> img_binary(number_of_colors);
    for(int i = 0; i < number_of_colors; ++i)
    {        
        cv::inRange(img_hsv, cv::Scalar(hsv_params_.H_min[i], hsv_params_.S_min[i], hsv_params_.V_min[i]),
                             cv::Scalar(hsv_params_.H_max[i], hsv_params_.S_max[i], hsv_params_.V_max[i]),
                    img_binary[i]);

        if(hsv_params_.display[i])
        {
            std::stringstream ss;
            ss << "Mask "<< i;
            cv::imshow(ss.str(), img_binary[i]);
            cv::waitKey(1);
        }
    }

    // ** Remove noise with erosion and dilation
    double max_pixel_count = 0;
    int max_idx = 0;
    for(std::size_t i = 0; i < img_binary.size(); ++i)
    {
        cv::dilate(img_binary[i],img_binary[i],cv::Mat(),cv::Point(-1,-1),dilate_times); //dilate image1 to fill empty spaces)
        cv::erode(img_binary[i],img_binary[i],cv::Mat(),cv::Point(-1,-1),erode_times); //erode image1 to remove single pixels
        double pixel_count = cv::countNonZero(img_binary[i]);
        if(pixel_count > max_pixel_count)
        {
            max_pixel_count = pixel_count;
            max_idx = i;
        }
    }
    // ** Select mask with biggest pixel count
    cv::Mat &mask = img_binary[max_idx];

    // ** Find biggest contour
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask,contours,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    if(contours.size() > 0)
    {
        std::vector<cv::Point> biggest_contour;
        // ** Get biggest contour
        double max_size = -1;
        double max_i = 0;
        for(std::size_t s = 0; s < contours.size(); ++s)
        {
            double size = cv::contourArea(contours[s]);
            if(size > max_size)
            {
                max_size = size;
                max_i = s;
            }
        }

        // ** Filter by aspect ratio and size
        cv::Rect bbox = cv::boundingRect(contours[max_i]);
        double aspect_ratio = std::max(bbox.height/bbox.width, bbox.width/bbox.height);

        if (max_size > MIN_PIXEL_OBJECT && aspect_ratio < MAX_ASPECT_RATIO)
        {
            //** Create mask
            biggest_contour = contours[max_i];
            cv::drawContours(out_img,contours, max_i, cv::Scalar(255), CV_FILLED);

            // ** Get mass center
            cv::Moments m = cv::moments(biggest_contour);
            mass_center.x = m.m10/m.m00;
            mass_center.y = m.m01/m.m00;

            return max_idx;
        }
    }
    return -1;
}

bool Object_Detection::detectObstacle(const cv::Mat &floor_mask)
{
    for(unsigned int j=ROI_MIN_U+80; j < ROI_MAX_U-80; j+= 10)
    {
        bool lineGood = true;
        for(unsigned int i=ROI_MIN_V; i < ROI_MAX_V-100; i+=5)
        {
            if(floor_mask.at<uint8_t>(i,j) == 0) // there is visible floor
            {
                lineGood = false;
                break; //Move to next column
            }
        }
        if(lineGood)
            return true;
    }
    return false;
}

bool Object_Detection::is_concave(const cv::Mat &depth_img, const cv::Mat &mask_img)
{
    // ** Shrink mask in order not to see the remaining floor
    cv::Mat mask_img2;
    int size = 5; // This depends on scaling factor
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,cv::Size(2*size+1, 2*size+1),cv::Point( size, size) );
    cv::erode(mask_img, mask_img2, element);

    double nanPoints=0, normalPoints=0, totalPoints=0;
    for(unsigned int i = 0; i < depth_img.rows; i+=STEP)
    {
        for(unsigned int j = 0; j < depth_img.cols; j+=STEP)
        {
            if(mask_img2.at<uint8_t>(i,j) != 0)
            {
                float v = depth_img.at<float>(i,j);
                if(!isnan(v) && v!=0)
                    ++normalPoints;
                else
                    ++nanPoints;

                ++totalPoints;
            }
        }
    }
    double ratio = nanPoints/totalPoints;
   return ratio > 0.6;
}

void Object_Detection::DR_Callback(ColorTuningConfig &config, uint32_t level)
{
    ROS_INFO("[Object Detection] Dynamic reconfigure");
    hsv_params_.H_min = {config.R_H_min, config.G_H_min,config.B_H_min,config.Y_H_min,config.P_H_min};
    hsv_params_.H_max = {config.R_H_max, config.G_H_max,config.B_H_max,config.Y_H_max,config.P_H_max};
    hsv_params_.S_min = {config.R_S_min, config.G_S_min,config.B_S_min,config.Y_S_min,config.P_S_min};
    hsv_params_.S_max = {config.R_S_max, config.G_S_max,config.B_S_max,config.Y_S_max,config.P_S_max};
    hsv_params_.V_min = {config.R_V_min, config.G_V_min,config.B_V_min,config.Y_V_min,config.P_V_min};
    hsv_params_.V_max = {config.R_V_max, config.G_V_max,config.B_V_max,config.Y_V_max,config.P_V_max};

    hsv_params_.display = {config.R_Display, config.G_Display, config.B_Display, config.Y_Display, config.P_Display};

    hsv_params_.remove_floor = config.Remove_floor;
}

void Object_Detection::load_calibration(const std::string &path)
{
    std::ifstream file;
    file.open(path.c_str());
    t_cam_to_robot0_ = Eigen::Matrix4f::Identity();
    for(unsigned int i = 0; i < 4; ++i)
    {
        for(unsigned int j = 0; j < 4; ++j)
        {
            file >> t_cam_to_robot0_(i,j);
        }
    }
    t_cam_to_robot_ = t_cam_to_robot0_;
    t_robot_to_cam_ = t_cam_to_robot_.inverse();
    std::cout << "Read calibration: "<<std::endl << t_cam_to_robot_<<std::endl;
    file.close();
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

void Object_Detection::publish_obstacle(bool is_obstacle)
{
    std_msgs::Bool msg;
    msg.data = is_obstacle;
    obstacle_pub_.publish(msg);
}

void Object_Detection::publish_object(const std::string &id, const pcl::PointXY position)
{
    ROS_ERROR("TO DO");
}

void Object_Detection::IMUCallback(const sensor_msgs::ImuConstPtr &imu_msg)
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
    t_robot_to_cam_ = t_cam_to_robot_.inverse();
}

void Object_Detection::OdometryCallback(const geometry_msgs::Pose2DConstPtr &odometry_msg)
{
    double x,y,theta;
    x = odometry_msg->x;
    y = odometry_msg->y;
    theta = odometry_msg->theta;

    robot_pose_ << cos(theta), -sin(theta), x,
                   sin(theta),  cos(theta), y,
                            0,           0, 1.0;
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

bool Object_Detection::is_new_object(const pcl::PointXY &position)
{
    std::cout << "Is new object: "<<objects_position_.size() << "objects"<<std::endl;
    for(std::size_t i = 0; i < objects_position_.size(); ++i)
    {
        double d =PCL_Utils::euclideanDistance(position, objects_position_[i].position_);
        std::cout << "Distance to map: "<<d<<std::endl;
        if(d < NEW_OBJECT_MIN_DISTANCE)
            return false;
    }
    return true;
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
        auto visualize_type = i < objects_position_.size() ? visualization_msgs::Marker::CUBE : visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker_obj.header.frame_id = COORD_FRAME_WORLD;
        marker_obj.header.stamp = ros::Time();
        marker_obj.ns = "Objects";
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
        marker_obj.type = visualize_type;
        marker_obj.text = obj.id_;

    }
    marker_pub_.publish( msg);
}

}  // namespace
