#include <object_detection/object_detection.h>

namespace object_detection
{
Object_Detection::Object_Detection(const ros::NodeHandle& n,
                                   const ros::NodeHandle& n_private)
    : n_(n), n_private_(n_private), frame_counter_(0), started_(false), it_(n), detected_object_(false)
{
    // ** Load camera extrinsic calibration
    load_calibration(RAS_Names::CALIBRATION_PATH);

    // ** Publishers
    image_pub_ = it_.advertise("/object_detection/image", 1);
    speaker_pub_ = n_.advertise<std_msgs::String>("/espeak/string", 1000);

    // ** Subscribers
    rgb_sub_.subscribe(n_, "/camera/rgb/image_rect_color", QUEUE_SIZE);
    depth_sub_.subscribe(n_, "/camera/depth_registered/hw_registered/image_rect_raw",QUEUE_SIZE);

    rgbd_sync_.reset(new RGBD_Sync(RGBD_Sync_Policy(QUEUE_SIZE), rgb_sub_, depth_sub_));
    rgbd_sync_->registerCallback(boost::bind(&Object_Detection::RGBD_Callback, this, _1, _2));

    // ** Services
    service_client_ = n_.serviceClient<ras_srv_msgs::Recognition>("/object_recognition/recognition");

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
        cv::Mat rgb_scaled, depth_scaled;

        // ** Create point cloud and transform into robot coordinates
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        buildPointCloud(rgb_img, depth_img, cloud);

        // ** Remove main plane (floor)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        get_floor_plane(cloud, floor_cloud);

        // ** Back-project to get binary mask
        cv::Mat floor_mask = 255*cv::Mat::ones(rgb_img.rows, rgb_img.cols, CV_8UC1);
        backproject_floor(floor_cloud, floor_mask);

        cv::bitwise_and(floor_mask,ROI_, floor_mask); //Combine with ROI
        cv::imshow("Floor mask",floor_mask);
        cv::waitKey(1);

        // ** Apply mask to remove floor in 2D
        cv::Mat rgb_filtered;
        rgb_img.copyTo(rgb_filtered, floor_mask);

        cv::imshow("Filtered image",rgb_filtered);
        cv::waitKey(1);

        // ** Color analysis (Ryan)
        cv::Point2i mass_center;
        cv::Mat out_image;
        int color = color_analysis(rgb_filtered, mass_center, out_image);

        cv::imshow("Color mask", out_image);
        cv::waitKey(1);

        // ** Call the recognition node if object found
        if (color >= 0)
        {
            if(!detected_object_)
            {
                detected_object_ = true;
                // ** Publish to speaker
                std_msgs::String msg;
                msg.data = "I see an object";
                ROS_WARN("OBJECT DETECTED");
//                speaker_pub_.publish(msg);

                /// @todo contact brain node in order to switch the navigation mode: move towards robot
            }

//             ** Call recognition if close enough
            if(mass_center.y > MIN_MASS_CENTER_Y) // If the object is close enough
            {
//                /@todo contact the brain node so that it can stop the robot
//                /

//                 Convert to ROS msg
                cv_bridge::CvImage out_msg;
                out_msg.header = rgb_msg->header;
                out_msg.encoding = sensor_msgs::image_encodings::MONO8;
                out_msg.image = out_image;

                // Publish to Services
                image_pub_.publish(rgb_msg); // Republish RGB image

                // Call recognition service
                ras_srv_msgs::Recognition srv;

                srv.request.rgb_img = *rgb_msg;
                srv.request.mask    = *out_msg.toImageMsg();
                srv.request.color   = color;
                if (service_client_.call(srv))
                {
                    detected_object_ = false;
                }
                else{
                    ROS_ERROR("Failed to call recognition service");
                }
            }
            else{
//                / @todo publish mass_center
            }
        }
    }
    else
        frame_counter_++;

    ros::WallTime t_end = ros::WallTime::now();
    ROS_INFO("[Object Detection] %.3f ms", RAS_Utils::time_diff_ms(t_begin, t_end));
}

void Object_Detection::buildPointCloud(const cv::Mat &rgb_img,
                                       const cv::Mat &depth_img,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_out)
{    
    int height = rgb_img.rows * SCALE_FACTOR;
    int width  = rgb_img.cols * SCALE_FACTOR;
    int step = 1.0 / SCALE_FACTOR;

    // ** Construct point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->resize(height*width);

    int i = 0;
    for (unsigned int v = 0; v < rgb_img.rows; v+=step)
    {
        for (unsigned int u = 0; u < rgb_img.cols; u+=step)
        {
            float z_m = depth_img.at<float>(v, u);

            const cv::Vec3b& c = rgb_img.at<cv::Vec3b>(v, u);

            pcl::PointXYZRGB& pt = cloud->points[i++];

            pt.x = z_m * ((u - CX) * FX_INV);
            pt.y = z_m * ((v - CY) * FY_INV);
            pt.z = z_m;

            pt.r = c[0];
            pt.g = c[1];
            pt.b = c[2];
        }
    }
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = true;

    // ** Transform into robot coordinate frame
    pcl::transformPointCloud(*cloud, *cloud_out, t_cam_to_robot_);
}

void Object_Detection::get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-1.0, 0.015);
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
    int size = 2; // This depends on scaling factor
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
    int pixel_threshhold=200;
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

    for(int i = 0; i < number_of_colors; ++i)
    {
        //cout<< "LOOP BEGIN!"<<endl;
        //waitKey(0); //wait infinite time for a keypress

//        i = i + 1;
        //cout << "value of i: " << i << endl;


        ////////////////////////////////////////////////////////////////////////////////////////////////
        //       3)Convert HSV image into Binary image by providing Higher and lower HSV ranges       //
        ////////////////////////////////////////////////////////////////////////////////////////////////

    cv::Mat img_binary = cv::Mat::zeros(img_hsv.rows, img_hsv.cols, CV_8UC1);

        switch (i)
        {
            case 0: // BLUE
                cv::inRange(img_hsv, cv::Scalar(hsv_params_.B_H_min, hsv_params_.B_S_min, hsv_params_.B_V_min),
                                     cv::Scalar(hsv_params_.B_H_max, hsv_params_.B_S_max, hsv_params_.B_V_max),
                            img_binary);
                if(hsv_params_.B_display){
                    cv::imshow("Blue mask", img_binary);
                    cv::waitKey(1);
                }
                break;
            case 1: // RED
            cv::inRange(img_hsv, cv::Scalar(hsv_params_.R_H_min, hsv_params_.R_S_min, hsv_params_.R_V_min),
                                 cv::Scalar(hsv_params_.R_H_max, hsv_params_.R_S_max, hsv_params_.R_V_max),
                        img_binary);
                if(hsv_params_.R_display){
                    cv::imshow("Red mask", img_binary);
                    cv::waitKey(1);
                }
            break;
            case 2: // GREEN
            cv::inRange(img_hsv, cv::Scalar(hsv_params_.G_H_min, hsv_params_.G_S_min, hsv_params_.G_V_min),
                                 cv::Scalar(hsv_params_.G_H_max, hsv_params_.G_S_max, hsv_params_.G_V_max),
                        img_binary);
                if(hsv_params_.G_display){
                    cv::imshow("Green mask", img_binary);
                    cv::waitKey(1);
                }
                break;
            case 3: // YELLOW
            cv::inRange(img_hsv, cv::Scalar(hsv_params_.Y_H_min, hsv_params_.Y_S_min, hsv_params_.Y_V_min),
                                 cv::Scalar(hsv_params_.Y_H_max, hsv_params_.Y_S_max, hsv_params_.Y_V_max),
                        img_binary);
                if(hsv_params_.Y_display){
                    cv::imshow("Yellow mask", img_binary);
                    cv::waitKey(1);
                }
                break;
            case 4: // PURPLE
            cv::inRange(img_hsv, cv::Scalar(hsv_params_.P_H_min, hsv_params_.P_S_min, hsv_params_.P_V_min),
                                 cv::Scalar(hsv_params_.P_H_max, hsv_params_.P_S_max, hsv_params_.P_V_max),
                        img_binary);
                if(hsv_params_.P_display){
                    cv::imshow("Purple mask", img_binary);
                    cv::waitKey(1);
                }
            break;
        }

//        cv::imshow("win2",img_binary);
//        cv::waitKey(1);

//        ////////////////////////////////////////////////
//        //       4) dilate then erode IMAGE           //
//        ////////////////////////////////////////////////
//        cv::dilate(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1),dilate_times); //dilate image1 to fill empty spaces)
////        cv::imshow("After dilate", img_binary); //display erode image
//        cv::erode(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1),erode_times); //erode image1 to remove single pixels
////        cv::imshow("After erode", img_binary); //display erode image
////        cv::waitKey(1);

//        ////////////////////////////////////////
//        //       5) FIND CONTOUR              //
//        ////////////////////////////////////////
//        std::vector<std::vector<cv::Point> > contours;
//        cv::findContours(img_binary,contours,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//        if(contours.size() > 0)
//        {
//            std::vector<cv::Point> biggest_contour;

//            // ** Get biggest contour
//            double max_size = -1;
//            double max_i = 0;
//            for(std::size_t s = 0; s < contours.size(); ++s)
//            {
//                double size = cv::contourArea(contours[s]);
//                if(size > max_size)
//                {
//                    max_size = size;
//                    max_i = s;
//                }
//            }
//            ///////////////////////////////////////////////////////////////////////////////////
//            //        6) Check if the object is found and break the loop accordingly         //
//            ///////////////////////////////////////////////////////////////////////////////////

//            if (max_size > pixel_threshhold)
//            {
//                // ** Get mass center
//                biggest_contour = contours[max_i];
//                cv::Moments mu = moments(biggest_contour,false);
//                mass_center.x = (mu.m10/mu.m00) / SCALE_FACTOR;
//                mass_center.y = (mu.m01/mu.m00) / SCALE_FACTOR;
//                //** Create image to publish
//                cv::drawContours(out_img,contours, max_i, cv::Scalar(255), CV_FILLED);
//                cv::resize(out_img, out_img, cv::Size(0,0), 1.0/SCALE_FACTOR, 1.0/SCALE_FACTOR); // So that we can get more detail
////                return i;
//            }
//        }
    }
    return -1;
}

void Object_Detection::DR_Callback(ColorTuningConfig &config, uint32_t level)
{
    ROS_INFO("[Object Detection] Dynamic reconfigure");
    hsv_params_.R_H_min = config.R_H_min;
    hsv_params_.R_H_max = config.R_H_max;
    hsv_params_.R_S_min = config.R_S_min;
    hsv_params_.R_S_max = config.R_S_max;
    hsv_params_.R_V_min = config.R_V_min;
    hsv_params_.R_V_max = config.R_V_max;
    hsv_params_.R_display = config.R_Display;

    hsv_params_.G_H_min = config.G_H_min;
    hsv_params_.G_H_max = config.G_H_max;
    hsv_params_.G_S_min = config.G_S_min;
    hsv_params_.G_S_max = config.G_S_max;
    hsv_params_.G_V_min = config.G_V_min;
    hsv_params_.G_V_max = config.G_V_max;
    hsv_params_.G_display = config.G_Display;

    hsv_params_.B_H_min = config.B_H_min;
    hsv_params_.B_H_max = config.B_H_max;
    hsv_params_.B_S_min = config.B_S_min;
    hsv_params_.B_S_max = config.B_S_max;
    hsv_params_.B_V_min = config.B_V_min;
    hsv_params_.B_V_max = config.B_V_max;
    hsv_params_.B_display = config.B_Display;

    hsv_params_.Y_H_min = config.Y_H_min;
    hsv_params_.Y_H_max = config.Y_H_max;
    hsv_params_.Y_S_min = config.Y_S_min;
    hsv_params_.Y_S_max = config.Y_S_max;
    hsv_params_.Y_V_min = config.Y_V_min;
    hsv_params_.Y_V_max = config.Y_V_max;
    hsv_params_.Y_display = config.Y_Display;

    hsv_params_.P_H_min = config.P_H_min;
    hsv_params_.P_H_max = config.P_H_max;
    hsv_params_.P_S_min = config.P_S_min;
    hsv_params_.P_S_max = config.P_S_max;
    hsv_params_.P_V_min = config.P_V_min;
    hsv_params_.P_V_max = config.P_V_max;
    hsv_params_.P_display = config.P_Display;

    hsv_params_.remove_floor = config.Remove_floor;
}

void Object_Detection::load_calibration(const std::string &path)
{
    std::ifstream file;
    file.open(path.c_str());
    t_cam_to_robot_ = Eigen::Matrix4f::Identity();
    for(unsigned int i = 0; i < 4; ++i)
    {
        for(unsigned int j = 0; j < 4; ++j)
        {
            file >> t_cam_to_robot_(i,j);
        }
    }
    t_robot_to_cam_ = t_cam_to_robot_.inverse();
    std::cout << "Read calibration: "<<std::endl << t_cam_to_robot_<<std::endl;
    file.close();
}

}  // namespace
