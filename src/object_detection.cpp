#include <object_detection/object_detection.h>

namespace object_detection
{
Object_Detection::Object_Detection(const ros::NodeHandle& n,
                                   const ros::NodeHandle& n_private)
    : n_(n), n_private_(n_private), frame_counter_(0), started_(false), it_(n), detected_object_(false)
{
    // ** Publishers
    image_pub_ = it_.advertise("/object_detection/image", 1);
    speaker_pub_ = n_.advertise<std_msgs::String>("/espeak/string", 1000);

    // ** Subscribers
    rgb_sub_.subscribe(n_, "/camera/rgb/image_raw", QUEUE_SIZE);
    depth_sub_.subscribe(n_, "/camera/depth_registered/image_raw",QUEUE_SIZE);

    rgbd_sync_.reset(new RGBD_Sync(RGBD_Sync_Policy(QUEUE_SIZE), rgb_sub_, depth_sub_));
    rgbd_sync_->registerCallback(boost::bind(&Object_Detection::RGBD_Callback, this, _1, _2));

    // ** Services
    service_client_ = n_.serviceClient<ras_srv_msgs::Recognition>("/object_recognition/recognition");

    // ** Create ROI
    ROI_ = cv::Mat::zeros(IMG_ROWS*SCALE_FACTOR, IMG_COLS*SCALE_FACTOR, CV_8UC1);
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
    if(frame_counter_ > 100 || started_)
    {
        started_ = true;
        // ** Convert ROS messages to OpenCV images and scale
        cv_bridge::CvImageConstPtr rgb_ptr   = cv_bridge::toCvShare(rgb_msg);
        cv_bridge::CvImageConstPtr depth_ptr   = cv_bridge::toCvShare(depth_msg);

        const cv::Mat& rgb_img     = rgb_ptr->image;
        const cv::Mat& depth_img   = depth_ptr->image;

        cv::Mat rgb_img_scaled, depth_img_scaled;
        cv::resize(rgb_img, rgb_img_scaled, cv::Size(0,0), SCALE_FACTOR, SCALE_FACTOR);
        cv::resize(depth_img, depth_img_scaled, cv::Size(0,0), SCALE_FACTOR, SCALE_FACTOR);
        cv::imshow("RGB", rgb_img_scaled);
        cv::waitKey(1);

        // ** Create point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        ros::WallTime t1 = ros::WallTime::now();

        buildPointCloud(rgb_img_scaled, depth_img_scaled, *cloud);

        ros::WallTime t2 = ros::WallTime::now();
        double t_build_pcl = RAS_Utils::time_diff_ms(t1,t2);

//        pcl::io::savePCDFile("/home/carlos/test_pcl.pcd", *cloud);

        // ** Remove main plane (floor)
        t1 = ros::WallTime::now();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr floor_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        get_floor_plane(cloud, floor_cloud);

        t2 = ros::WallTime::now();
        double t_remove_plane = RAS_Utils::time_diff_ms(t1,t2);
//        pcl::io::savePCDFile("/home/carlos/test_pcl.pcd", *floor_cloud);

        // ** Back-project to get binary mask
        t1 = ros::WallTime::now();
        cv::Mat floor_mask = 255*cv::Mat::ones(rgb_img_scaled.rows, rgb_img_scaled.cols, CV_8UC1);

        backproject_floor(floor_cloud, floor_mask);

        t2 = ros::WallTime::now();
        double t_backproject = RAS_Utils::time_diff_ms(t1,t2);
        cv::bitwise_and(floor_mask,ROI_, floor_mask); //Combine with ROI
        cv::imshow("Floor mask",floor_mask);
        cv::waitKey(1);

        // ** Apply mask to remove floor in 2D
        cv::Mat rgb_filtered; //= cv::Mat::zeros(rgb_img.rows, rgb_img.cols, CV_8UC3);
        rgb_img_scaled.copyTo(rgb_filtered, floor_mask);
        cv::imshow("Filtered image",rgb_filtered);
        cv::waitKey(1);

        // ** Color analysis (Ryan)
        t1 = ros::WallTime::now();
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

            // ** Call recognition if close enough
//            if(mass_center.y > MIN_MASS_CENTER_Y) // If the object is close enough
//            {
                ///@todo contact the brain node so that it can stop the robot
                ///

                // Convert to ROS msg
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
//            }
//            else{
                /// @todo publish mass_center
//            }
        }


        // ** Print statistics
//        ROS_INFO("[Object Detection] Build PCL: %.3f,  Plane: %.3f, Backproject: %.3f, Color: %.3f, TOTAL: %.3f ms",
//                 t_build_pcl, t_remove_plane, t_backproject, t_color, t_total);
    }
    else
        frame_counter_++;

    ros::WallTime t_end = ros::WallTime::now();
//    ROS_INFO("[Object Detection] %.3f ms", RAS_Utils::time_diff_ms(t_begin, t_end));
}

void Object_Detection::buildPointCloud(const cv::Mat &rgb_img,
                                       const cv::Mat &depth_img,
                                             pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    int height = rgb_img.rows;
    int width  = rgb_img.cols;

    // ** Construct point cloud
    cloud.resize(height*width);

    for (unsigned int u = 0; u < width; ++u)
    {
        for (unsigned int v = 0; v < height; ++v)
        {
            float z_m = depth_img.at<float>(v, u);

            const cv::Vec3b& c = rgb_img.at<cv::Vec3b>(v, u);

            pcl::PointXYZRGB& pt = cloud.points[v*width + u];

            if (z_m != 0)
            {
                pt.x = z_m * ((u - CX) * FX_INV);
                pt.y = z_m * ((v - CY) * FY_INV);
                pt.z = z_m;

                pt.r = c[0];
                pt.g = c[1];
                pt.b = c[2];
            }
            else
            {
                pt.x = pt.y = pt.z = 0;//std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    cloud.width = width;
    cloud.height = height;
    cloud.is_dense = true;
}

void Object_Detection::get_floor_plane(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud_in,
                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr      &cloud_out)
{
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.005);

    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_in);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;

    // Extract the object
    extract.setInputCloud (cloud_in);
    extract.setIndices (inliers);
    extract.setNegative (false); //Extract the plane!
    extract.filter (*cloud_out);
}

void Object_Detection::backproject_floor(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &floor_cloud,
                                                                                    cv::Mat &floor_mask)
{
    for(std::size_t i = 0; i < floor_cloud->size(); ++i)
    {
        const pcl::PointXYZRGB& pt = floor_cloud->points[i];
        if(pt.z != 0)
        {
            int u = pt.x * FX / (pt.z) + CX;
            int v = pt.y * FY / (pt.z) + CY;
            floor_mask.at<uint8_t>(v,u) = 0;
        }
    }

    // ** Opening (dilate + erosion)
    int size = 1;
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

    int hue_low_blue =5; int hue_low_red =105; int hue_low_green =50; int hue_low_purple =140; int hue_low_yellow =85;
    int hue_high_blue=25; int hue_high_red=130; int hue_high_green=85; int hue_high_purple=170; int hue_high_yellow=105;
    //Hue values: Blue(15-25), Red(115-120), Green(55-65), Purple(140-170), Orange(110-120)
    //Perhaps to use saturation and brightness to distinguish orange with red, or to increase contrast of input image
    int sat_low=1;
    int sat_high=255;
    int bright_low=1;
    int bright_high=255;

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
            case 0:
                cv::inRange(img_hsv, cv::Scalar(hue_low_blue, sat_low, bright_low),
                                     cv::Scalar(hue_high_blue, sat_high, bright_high),
                            img_binary);
                break;
            case 1:
                cv::inRange(img_hsv, cv::Scalar(hue_low_red, 100, bright_low),
                                 cv::Scalar(hue_high_red, 255, bright_high),
                        img_binary);
                break;
            case 2:

                cv::inRange(img_hsv, cv::Scalar(hue_low_green, sat_low, bright_low),
                                     cv::Scalar(hue_high_green, sat_high, bright_high),
                            img_binary);
                break;
            case 3:
                cv::inRange(img_hsv, cv::Scalar(hue_low_purple, sat_low, bright_low),
                                     cv::Scalar(hue_high_purple, sat_high, bright_high),
                            img_binary);
                break;
            case 4:
                cv::inRange(img_hsv, cv::Scalar(hue_low_yellow, 100, bright_low),
                                     cv::Scalar(hue_high_yellow, 255, bright_high),
                            img_binary);
                break;
        }
        cv::imshow("win2",img_binary);
        cv::waitKey(1);


        ////////////////////////////////////////////////
        //       4) dilate then erode IMAGE           //
        ////////////////////////////////////////////////
        cv::dilate(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1),dilate_times); //dilate image1 to fill empty spaces)
//        cv::imshow("After dilate", img_binary); //display erode image
        cv::erode(img_binary,img_binary,cv::Mat(),cv::Point(-1,-1),erode_times); //erode image1 to remove single pixels
//        cv::imshow("After erode", img_binary); //display erode image
//        cv::waitKey(1);

        ////////////////////////////////////////
        //       5) FIND CONTOUR              //
        ////////////////////////////////////////
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(img_binary,contours,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
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

            if (max_size > pixel_threshhold)
            {
                // ** Get mass center
                biggest_contour = contours[max_i];
                cv::Moments mu = moments(biggest_contour,false);
                mass_center.x = (mu.m10/mu.m00) / SCALE_FACTOR;
                mass_center.y = (mu.m01/mu.m00) / SCALE_FACTOR;
                //** Create image to publish
                cv::drawContours(out_img,contours, max_i, cv::Scalar(255), CV_FILLED);
                cv::resize(out_img, out_img, cv::Size(0,0), 1.0/SCALE_FACTOR, 1.0/SCALE_FACTOR); // So that we can get more detail
                return i;
            }
        }
    }
    cv::resize(out_img, out_img, cv::Size(0,0), 1.0/SCALE_FACTOR, 1.0/SCALE_FACTOR); // So that we can get more detail
    return -1;
}

}  // namespace
