#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64.h"
#include "nav_msgs/Odometry.h"
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <crawler_msgs/TextureTriggerSrv.h>
#include <crawler_msgs/LiveTexture.h>
#include <crawler_msgs/PoseTriggerSrv.h>
//#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <image_transport/image_transport.h>
#include "vector"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <tf/transform_broadcaster.h>
#include <pcl/filters/conditional_removal.h>
using namespace std;
#define PI 3.14159265
class Mapper

{
protected:


    float priorDynamic;
    double lastTime, rotationTuning;
    int iteration;
    tf::Quaternion q0;
    tf::Vector3 t0;
    Eigen::Quaternion<double> rotation;
    Eigen::Matrix3f intrinsics;
    Eigen::MatrixXf extrinsics;
    //  nav_msgs::OccupancyGrid globalOutlierMap;  //obstacles occupancy grid
    geometry_msgs::PoseStamped poseStamped;
    nav_msgs::Path path;
    image_transport::Publisher obsImagePub;

    double yawInit=0;

    int erosion_elem = 0;
    int dilation_elem = 0;

    int erosion_size = 2;
    int dilation_size =4;

    bool getTexture, mergeTextures_, keepBackground_;
    std::string icpParamPath;
    std::string icpInputParamPath;
    std::string icpPostParamPath;
    bool init, useTf;
    bool computeProbDynamic;
    ros::NodeHandle n;
    ros::ServiceServer service, poseService;

    ros::Publisher inliersTexturePub, outliersTexturePub, mergedTexturePub;

    image_transport::ImageTransport it;

    message_filters::Subscriber<sensor_msgs::PointCloud2> inliersSub, outliersSub;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl_acc ;
    pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl_acc;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;
    tf::StampedTransform ifmTf, zeroTf;
    tf::TransformListener listener1, listener2;
    tf::TransformBroadcaster broadcaster;
    int rows, cols, fx, fy, cx, cy;
    cv::Mat texture;
    double resolution_;
    image_transport::Subscriber textureSub;

    Eigen::Matrix4d getPoseTf(){


        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();

        try{


            tf::StampedTransform tempTf;
            listener1.waitForTransform("odom_ekf", "base_link", ros::Time(0), ros::Duration(1.0) );

            listener1.lookupTransform("odom_ekf", "base_link", ros::Time(0), tempTf);

            tf::Quaternion q=tempTf.getRotation();
            tf::Vector3 t=tempTf.getOrigin();
            pose(0,3)=t.x();
            pose(1,3)=t.y();
            pose(2,3)=t.z();

            Eigen::Quaterniond qqq(q.w(),q.x(),q.y(),q.z());
            Eigen::Matrix3d mmm(qqq);

            pose.block(0,0,3,3)=mmm;
            //pose=pose.inverse().eval();
            // pose(0,3)=pose(0,3);
            //pose(1,3)=pose(1,3);

            //  std::cout<<"tf retrieved, ekf to base is \n" <<pose<<std::endl;


        }

        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Received an exception trying to transform: %s", ex.what());
        }

        //pose=zeroTransform*pose;

        //std::cout<<"tf retrieved, ekf to base is 2 \n" <<pose<<std::endl;

        return pose;

    }

    void publish(cv::Mat& data, ros::Publisher& pub, pcl::PCLHeader header, std::string frame_id, double resolution, Eigen::Matrix4d& pose){

        std_msgs::Header h;
        pcl_conversions::fromPCL(header, h);

        sensor_msgs::ImagePtr msg;
        msg= cv_bridge::CvImage(std_msgs::Header(),"bgr8", data).toImageMsg();

        msg->header.stamp=h.stamp;
        msg->header.frame_id="base_link";

        crawler_msgs::LiveTexture texture;
        //header.stamp=header.stamp;
        //header.seq=img_msg->header.seq;


        nav_msgs::MapMetaData info;
        Eigen::Matrix3d dcm;
        dcm=pose.block(0,0,3,3);
        Eigen::Quaterniond q(dcm);
        info.map_load_time=h.stamp;
        info.origin.position.x=pose(0,3);
        info.origin.position.y=pose(1,3);
        info.origin.position.z=pose(2,3);
        info.origin.orientation.x=q.x();
        info.origin.orientation.y=q.y();
        info.origin.orientation.z=q.z();
        info.origin.orientation.w=q.w();
        info.resolution=resolution;
        info.width=data.cols;
        info.height=data.rows;

        //                geometry_msgs::Pose Pose;
        //                info.Pose=Pose;

        texture.info=info;
        texture.header=h;
        texture.texture=*msg;
        texture.texture.header.seq=h.seq;
        pub.publish(texture);

    }
    void erode(  cv::Mat& src, unsigned int iterations=1 )
    {


        int erosion_type = 0;
        if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
        else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
        else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
        cv::Mat element = cv::getStructuringElement( erosion_type,
                                                     cv::Size( erosion_size + 1, erosion_size+1 ),
                                                     cv::Point( erosion_size, erosion_size ) );


        for (unsigned int i=0; i<iterations; i++){
            cv::erode( src, src, element );
        }
        //imshow( "Erosion Demo", src );
    }
    void dilate(cv::Mat& src, unsigned int iterations=1 )
    {
        int dilation_type = 0;
        if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
        else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
        else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
        cv::Mat element = cv::getStructuringElement( dilation_type,
                                                     cv::Size( dilation_size + 1, dilation_size+1 ),
                                                     cv::Point( dilation_size, dilation_size ) );

        for (unsigned int i=0; i<iterations; i++){
            cv::dilate( src, src, element );
        }
        //  imshow( "Dilation Demo", src );
    }
    void publish_colored_texture(cv::Mat texture, Eigen::Matrix4d& newPose, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, ros::Publisher& pub, std::string color, bool pubSwtich){


        cv::Size s=texture.size();
        pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy (new pcl::PointCloud<pcl::PointXY>);;
        std::cout<<"texture point cloud size is "<<cloud->size()<<std::endl;

        //   std::cout<<"pass -1"<<std::endl;
        pcl::copyPointCloud(*cloud, *cloud_xy);
        //       std::cout<<"pass -1.5"<<std::endl;
        int rows = s.height;
        int cols = s.width;

        double offset=(rows/2)*resolution_;

        //        for(int i=0;i<cloud->points.size();i++){

        //            pcl::PointXYZRGB p;
        //            p.x=cloud->points[i].x;
        //        }
        //    std::cout<<"pass -2"<<std::endl;

        pcl::KdTreeFLANN<pcl::PointXY> kdtree;
        kdtree.setInputCloud (cloud_xy);
        for (int v=0; v<rows; v++){
            for (int u=0; u<cols; u++)
            {
                pcl::PointXY searchPoint;
                searchPoint.x=double(rows-v)*resolution_;
                //  std::cout<<cols<<std::endl;
                searchPoint.y=double(cols/2-u)*resolution_;

                double correction=(rotationTuning*PI)/180;
                searchPoint.x=searchPoint.x*cos(correction)-searchPoint.y*sin(correction);
                searchPoint.y=searchPoint.x*sin(correction)+searchPoint.y*cos(correction);

                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;
                float radius = 0.01;

                if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
                {
                    //std::cout<<"it happened"<<std::endl;
                    if (color=="red"  && mergeTextures_){
                        texture.at<cv::Vec3b>(v,u)[2]=255;
                        texture.at<cv::Vec3b>(v,u)[1]=0;
                        texture.at<cv::Vec3b>(v,u)[0]=0;
                    }
                    if (color=="green" && mergeTextures_){
                        texture.at<cv::Vec3b>(v,u)[2]=0;
                        texture.at<cv::Vec3b>(v,u)[1]=255;
                        texture.at<cv::Vec3b>(v,u)[0]=0;
                    }
                    /* for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
                      std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x
                                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y
                                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z
                                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl; */
                }
                else
                {
                    if (!mergeTextures_){
                        texture.at<cv::Vec3b>(v,u)[2]=0;
                        texture.at<cv::Vec3b>(v,u)[1]=0;
                        texture.at<cv::Vec3b>(v,u)[0]=0;
                    }

                    int r= texture.at<cv::Vec3b>(v,u)[2];
                    int g= texture.at<cv::Vec3b>(v,u)[1];
                    int b= texture.at<cv::Vec3b>(v,u)[0];
                    bool unboserved=true;

                    if ( (r==255 && g==0 && b==0) || (r==0 && g==255 && b==0) || (r==0 && g==0 && b==0) ){

                        unboserved=false;
                    }

                    if (mergeTextures_ &&  unboserved  && !keepBackground_  )
                    {
                        texture.at<cv::Vec3b>(v,u)[2]=0;
                        texture.at<cv::Vec3b>(v,u)[1]=0;
                        texture.at<cv::Vec3b>(v,u)[0]=255;

                    }



                }

            }
        }

        erode(texture,1);
        dilate(texture,1);

        if (pubSwtich){
            publish(texture, pub, cloud->header ,cloud->header.frame_id, resolution_, newPose);
        }
        //std::cout<<texture.size()<<std::endl;
        //cv::imshow( "Undistorted", texture );
        //cv::waitKey(0);

    }



    void mapCb(const sensor_msgs::PointCloud2ConstPtr& inliers, const sensor_msgs::PointCloud2ConstPtr& outliers)
    {
        Eigen::Matrix4d pose=getPoseTf();
        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg( *inliers, *inliers_pcl);

        pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg( *outliers, *outliers_pcl);



        assert(poseStamped.header.stamp.toSec() >= lastTime);
        lastTime=poseStamped.header.stamp.toSec();
        //  pcl_conversions::toPCL(poseStamped.header.stamp, inliers_pcl->header.stamp);
        //  pcl_conversions::toPCL(poseStamped.header.stamp, outliers_pcl->header.stamp);

        Eigen::Matrix4d newPose=Eigen::Matrix4d::Identity();
        //geometry_msgs::PoseStamped poseStamped;


        newPose(0,3)=poseStamped.pose.position.x;
        newPose(1,3)=poseStamped.pose.position.y;
        newPose(2,3)=poseStamped.pose.position.z;


        Eigen::Quaterniond q;

        q.x()=poseStamped.pose.orientation.x;
        q.y()=poseStamped.pose.orientation.y;
        q.z()=poseStamped.pose.orientation.z;
        q.w()=poseStamped.pose.orientation.w;

        Eigen::Matrix3d dcm(q);
        newPose.block(0,0,3,3)=dcm;




        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_transformed (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_transformed (new pcl::PointCloud<pcl::PointXYZ>);

        pcl::transformPointCloud (*inliers, *inliers_transformed, poseStamped);
        pcl::transformPointCloud (*outliers, *outliers_transformed, poseStamped);

        inliers_pcl_acc=inliers_transformed+inliers_pcl_acc;
        outliers_pcl_acc=outliers_transformed+outliers_pcl_acc;


}

    void tempfn(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){


        if (!mergeTextures_)
        {
            cv::Mat texture_inliers=texture;
            cv::Mat texture_outliers=texture;

            if (inliers_pcl->size()>0){
                publish_colored_texture(texture_inliers,newPose,inliers_pcl, inliersTexturePub,"green", true);
            }


            if (outliers_pcl->size()>0){
                publish_colored_texture(texture_outliers,newPose,outliers_pcl, outliersTexturePub,"red", true);
            }

        }
        else{

            if (inliers_pcl->size()>0){
                publish_colored_texture(texture,newPose,inliers_pcl, mergedTexturePub,"green", true);
            }


            if (outliers_pcl->size()>0){
                publish_colored_texture(texture,newPose,outliers_pcl, mergedTexturePub,"red", true);
            }
             init=true;
        }








    }



    void img_callback(const sensor_msgs::ImageConstPtr & img_msg)  //ROS callback
    {


        if (init){
            Eigen::Matrix4d pose=getPoseTf();

            cv::Mat warp_mat(2, 3, CV_64F);
            Eigen::MatrixXd eigenRotation(2,3);
            eigenRotation(0,0)=pose(0,0);
            eigenRotation(0,1)=pose(0,1);

            eigenRotation(1,0)=pose(1,0);
            eigenRotation(1,1)=pose(1,1);


            eigenRotation(0,2)=pose(0,3);
            eigenRotation(1,2)=pose(1,3);

            cv::eigen2cv(eigenRotation, warp_mat);
            cv::Mat A(2, 3, CV_64F);
            cv::Mat warp_dst = cv::Mat::zeros( texture.rows, texture.cols, texture.type() );

            cv::warpAffine( texture, warp_dst, warp_mat, warp_dst.size() );



            publish(warp_dst, mergedTexturePub, pcl_conversions::toPCL(img_msg->header) ,img_msg->header.frame_id, resolution_, pose);
        }

        //std::cout<<"Cam2 Cb"<<std::endl;
        if (getTexture){
            std::cout<<"trigger ON..."<<std::endl;
            getTexture=false;
            try
            {
                cv::Mat img(cv_bridge::toCvCopy(img_msg,"bgr8")->image);

                //cv::Mat Undistortedimg;
                //cv::undistort(img, Undistortedimg, intrinsics2, distCoeffs2);

                //  cv::Size s=img.size();

                //rows = s.height;
                // cols = s.width;
                texture=img;

            }
            // cv::imshow( "Original", img );
            // cv::imshow( "Undistorted", Undistortedimg );
            //  cv::waitKey(0);

            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;

            }
        }
    }


public:

    bool add(crawler_msgs::TextureTriggerSrv::Request  &req, crawler_msgs::TextureTriggerSrv::Response &res){
        getTexture=req.data;
        std::cout<<"GOT SERVICE TEXTURE "<<std::endl;
        res.success = true;
        res.message= "done";
        return true;
    }
    bool gotpose(crawler_msgs::PoseTriggerSrv::Request  &req, crawler_msgs::PoseTriggerSrv::Response &res){

        poseStamped=req.data;
        std::cout<<"GOT POSE SERVICE "<<std::endl;
        res.success = true;
        res.message= "done";
        return true;
    }
    Mapper() : n("~"), it(n) {

        //  n.param<double>("x_offset", xOffset, 1);
        //  n.param<double>("y_offset", yOffset, 1);


        srand (time(NULL));
        ros::Duration(0.5).sleep();
        std::string transport = "raw";
        n.param("transport",transport,transport);
        //std::cout<<"ADVERTISED TEXTURE "<<std::endl;
        service= n.advertiseService("texture_trigger", &Mapper::add, this);
        poseService= n.advertiseService("pose_trigger", &Mapper::gotpose, this);
        lastTime=0;
        //      PoseTriggerSrv
        n.param<double>("resolution", resolution_, 0.01);
        n.param<bool>("merge_textures", mergeTextures_, false);
        n.param<bool>("keep_merged_texture_background", keepBackground_, false);
        n.param<double>("rotation_tuning", rotationTuning, -4);
        fx=608.151123046875; fy=606.9262084960938;
        cx=328.7842102050781; cy=242.90274047851562;
        intrinsics=Eigen::Matrix3f::Identity();
        intrinsics(0,0)=fx; intrinsics(0,2)=cx;
        intrinsics(1,1)=fy; intrinsics(1,2)=cy;
        intrinsics(2,2)=1;
        getTexture=false;

        init=false;

        inliersSub.subscribe(n, "inliers", 1);
        outliersSub.subscribe(n, "outliers", 1);

        //coloredScanPublisher = n.advertise<pcl::PointCloud<pcl::PointXYZRGBAI> > ("livox/lidar_colored", 1);

        inliersTexturePub = n.advertise<crawler_msgs::LiveTexture> ("texture_inliers", 1);
        outliersTexturePub = n.advertise<crawler_msgs::LiveTexture> ("texture_outliers", 1);

        if (mergeTextures_){

            mergedTexturePub = n.advertise<crawler_msgs::LiveTexture> ("texture_merged", 1);

        }
        // scan_sub_ = nh_.subscribe("inliers",1,&Mapper::inliers_callback,this);

        //encoder_sub_ = n.subscribe("encoder_sub",1,&Mapper::rolling_callback,this);
        textureSub=it.subscribe("/texture_in",1,&Mapper::img_callback,this,transport);
        // inliers_texture = it.advertise("obs_img", 1);

        sync.reset(new Sync(MySyncPolicy(10), inliersSub, outliersSub));

        sync->registerCallback(boost::bind(&Mapper::mapCb, this, _1, _2));
    }
};

int main(int argc, char * argv[]){


    ros::init(argc, argv, "crawler_texture_layers");
    Mapper var;


    ros::spin();

}
