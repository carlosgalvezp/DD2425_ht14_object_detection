#include <object_detection/object_detection.h>

namespace object_detection
{
Object_Detection::Object_Detection()
    : frame_counter_(0), started_(false), detected_object_(false), imu_calibrated_(false)
{
    // ** Load camera extrinsic calibration
    load_calibration(RAS_Names::CALIBRATION_PATH);

    // ** Publishers
    evidence_pub_ = n.advertise<ras_msgs::RAS_Evidence>(TOPIC_EVIDENCE, 10);
    speaker_pub_ = n.advertise<std_msgs::String>(TOPIC_SPEAKER, 10);
    obstacle_pub_ = n.advertise<std_msgs::Bool>(TOPIC_OBSTACLE, 10);

    // ** Subscribers
    imu_sub_ = n.subscribe(TOPIC_IMU, QUEUE_SIZE,  &Object_Detection::IMUCallback, this);

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
    ros_time = ros::WallTime::now();
}


void Object_Detection::RGBD_Callback(const sensor_msgs::ImageConstPtr &rgb_msg,
                                     const sensor_msgs::ImageConstPtr &depth_msg)
{
    ros::WallTime t_begin = ros::WallTime::now();
    if(frame_counter_ > 20 || started_)
    {
        started_ = true;
        // ** Convert ROS messages to OpenCV images and scale
        cv_bridge::CvImageConstPtr rgb_ptr   = cv_bridge::toCvShare(rgb_msg);
        cv_bridge::CvImageConstPtr depth_ptr   = cv_bridge::toCvShare(depth_msg);
        const cv::Mat& rgb_img     = rgb_ptr->image;
        const cv::Mat& depth_img   = depth_ptr->image;

        cv::imshow("BGR", rgb_img);
        // ** Detect color in the image
        cv::Mat color_mask = cv::Mat::zeros(rgb_img.rows, rgb_img.cols, CV_8UC1);
        pcl::PointXYZ position;
        pcl::PointCloud<pcl::PointXYZRGB>:: Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        int color = image_analysis(rgb_img, depth_img, color_mask, position, cloud);

        // ** Call the recognition node if object found
        if (color >= 0)
        {
            if(!detected_object_)
            {
                detected_object_ = true;
                ROS_INFO("Object detected");
                /// @todo contact brain node in order to switch the navigation mode: move towards robot
            }
            // ** Transform point into robot frame
            PCL_Utils::transformPoint(position, t_cam_to_robot_, position);
            std::cout << "Position: "<<position.x<<std::endl;
            // ** Call recognition if close enough
//            if(position.x < MIN_DIST_OBJECT)
//            {
                // ** Call navigation node to stop the robot
//                communicate(SRV_NAVIGATION_IN, RAS_Names::Navigation_Modes::NAVIGATION_STOP);

                // ** Wait a bit for the robot to stop
//                ros::Rate r(0.5); // [s]
//                r.sleep();

                // ** Recognition
                std::string object = object_recognition_.classify(rgb_img, depth_img, cloud,color_mask, t_cam_to_robot_, position);
                std::string msg = "I see a " + object;
                ROS_INFO("%s", msg.c_str());

                // ** Publish evidence and obstacle detection
                publish_evidence(object, rgb_img);
                publish_obstacle();
//            }
//            else{
                // Do something else?
//            }
        }
        // Leverage the work to detect obstacles
        else
        {
            if(detectObstacle(cloud))
                publish_obstacle();
        }
    }
    else
        frame_counter_++;

    ros::WallTime t_end = ros::WallTime::now();
    ROS_INFO("[Object Detection] %.3f ms", RAS_Utils::time_diff_ms(t_begin, t_end));
}

int Object_Detection::image_analysis(const cv::Mat &rgb_img, const cv::Mat &depth_img,
                                      cv::Mat &color_mask, pcl::PointXYZ &position, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    // ** Create point cloud and transform into robot coordinates
    PCL_Utils::buildPointCloud(rgb_img, depth_img, cloud, SCALE_FACTOR);
    pcl::transformPointCloud(*cloud, *cloud, t_cam_to_robot_);
//    PCL_Utils::visualizePointCloud(cloud);

    // ** Remove floor
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    get_floor_plane(cloud, floor_cloud);

    // ** Back-project to get binary mask
    cv::Mat floor_mask = 255*cv::Mat::ones(rgb_img.rows, rgb_img.cols, CV_8UC1);
    backproject_floor(floor_cloud, floor_mask);

    cv::bitwise_and(floor_mask, ROI_, floor_mask); //Combine with ROI
    cv::imshow("Floor mask",floor_mask);
    cv::waitKey(1);

    // ** Apply mask to remove floor in 2D
    cv::Mat rgb_filtered;
    rgb_img.copyTo(rgb_filtered, floor_mask);

    cv::imshow("Filtered image",rgb_filtered);
    cv::waitKey(1);

    // ** Color analysis (Ryan)
    cv::Point2i mass_center;
    int color = color_analysis(rgb_filtered, mass_center, color_mask);

    // ** Combine with 3D mask
    cv::bitwise_and(color_mask, floor_mask, color_mask);
    cv::imshow("Color mask", color_mask);
    cv::waitKey(1);

    // ** Get 3D position of the object, if any
    if(color >=0)
    {
        double depth = depth_img.at<float>(mass_center.y,mass_center.x);
        if(depth == 0 || isnan(depth))
        {
            depth = estimateDepth(depth_img, mass_center.y);
        }
        PCL_Utils::transform2Dto3D(mass_center, depth, position);
    }
    return color;
}

void Object_Detection::get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-1.0, 0.01);
    pass.filter (*cloud_out);
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
    int size = 4; // This depends on scaling factor
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,cv::Size(2*size+1, 2*size+1),cv::Point( size, size) );
    cv::erode(floor_mask, floor_mask, element);
    cv::dilate(floor_mask, floor_mask, element);
}

int Object_Detection::color_analysis(const cv::Mat &img,
                                           cv::Point2i &mass_center,
                                           cv::Mat &out_img)
{
    ////////////////////////////////////////////
    //             Parameters                 //
    ////////////////////////////////////////////

    int number_of_colors = 5; //blue,red,green,purple,yellow
    int erode_times=1;
    int dilate_times=1;

    out_img = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    ////////////////////////////////////////////////
    //       2) CONVERT BGR IMAGE TO HSV IMAGE    //
    ////////////////////////////////////////////////

    cv::Mat img_hsv;
    cv::cvtColor(img,img_hsv,CV_BGR2HSV);


    /////////////////////
    //   LOOP START    //
    /////////////////////
    std::vector<cv::Mat> img_binary(number_of_colors);
    for(int i = 0; i < number_of_colors; ++i)
    {        
        ////////////////////////////////////////////////////////////////////////////////////////////////
        //       3)Convert HSV image into Binary image by providing Higher and lower HSV ranges       //
        ////////////////////////////////////////////////////////////////////////////////////////////////
        cv::inRange(img_hsv, cv::Scalar(hsv_params_.H_min[i], hsv_params_.S_min[i], hsv_params_.V_min[i]),
                    cv::Scalar(hsv_params_.H_max[i], hsv_params_.S_max[i], hsv_params_.V_max[i]),
                    img_binary[i]);
        if(hsv_params_.display[i]){
            std::stringstream ss;
            ss << "Mask "<< i;
            cv::imshow(ss.str(), img_binary[i]);
            cv::waitKey(1);
        }
    }

    ////////////////////////////////////////////////
    //       4) dilate then erode IMAGE           //
    ////////////////////////////////////////////////
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
    cv::Mat &mask = img_binary[max_idx];

    ////////////////////////////////////////
    //       5) FIND CONTOUR              //
    ////////////////////////////////////////
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
        ///////////////////////////////////////////////////////////////////////////////////
        //        6) Check if the object is found and break the loop accordingly         //
        ///////////////////////////////////////////////////////////////////////////////////

        if (max_size > MIN_PIXEL_OBJECT)
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

bool Object_Detection::detectObstacle(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in)
{
    // ** Pass-through
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.03, 0.3);
    pass.filter (*cloud_filtered);

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.0, MIN_DIST_OBSTACLE);
    pass.filter (*cloud_filtered);

    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-ROBOT_BORDER, ROBOT_BORDER);
    pass.filter (*cloud_filtered);

    // ** If any point remaining, then we have an obstacle
    return cloud_filtered ->points.size() > 0;
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
    rgb_img.encoding = sensor_msgs::image_encodings::BGR8;
    rgb_img.image = image;

    // ** Create messages
    ras_msgs::RAS_Evidence msg;
    msg.stamp = stamp;
    msg.group_number = GROUP_NUMBER;
    msg.image_evidence = *rgb_img.toImageMsg();

    std_msgs::String speaker_msg;
    speaker_msg.data = "I see " + object_id;

    evidence_pub_.publish(msg);
    speaker_pub_.publish(speaker_msg);
}

void Object_Detection::publish_obstacle()
{
    std_msgs::Bool msg;
    msg.data = true;
    obstacle_pub_.publish(msg);
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

double Object_Detection::estimateDepth(const cv::Mat &depth_img, int y_coordinate)
{
    double depth=0;
    int count=0;
    for(unsigned int i = 0; i < IMG_COLS; ++i)
    {
        double d = depth_img.at<float>(y_coordinate,i);
        if( d!= 0 && !isnan(d))
        {
            depth += d;
            ++count;
        }
    }
    return depth/count;
}

}  // namespace
