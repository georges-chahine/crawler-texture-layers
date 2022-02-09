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
#include <pcl/octree/octree_search.h>
#include <crawler_msgs/TextureTriggerSrv.h>
#include <live_texture_msgs/LiveTexture.h>
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

    bool getTexture, mergeTextures_, keepBackground_, low_level_layer_;
    std::string icpParamPath;
    std::string icpInputParamPath;
    std::string icpPostParamPath;
    bool init, useTf;
    bool computeProbDynamic;
    ros::NodeHandle n;
    ros::ServiceServer service, poseService;

    ros::Publisher inliersTexturePub, outliersTexturePub, mergedTexturePub;
    ros::Subscriber odomSub;
    image_transport::ImageTransport it;

    message_filters::Subscriber<sensor_msgs::PointCloud2> inliersSub, outliersSub, inliersSubTmp, outliersSubTmp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl_acc, inliers_pcl_acc_tmp ;
    pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl_acc,  outliers_pcl_acc_tmp;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy, MySyncPolicyTmp;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync, SyncTmp;
    boost::shared_ptr<Sync> sync, syncTmp;
    tf::StampedTransform ifmTf, zeroTf;
    tf::TransformListener listener1, listener2;
    tf::TransformBroadcaster broadcaster;
    int rows, cols, fx, fy, cx, cy;
    cv::Mat texture, texture_temp, texture_temp2, texture_scan;
    double resolution_;
    double lastYaw, lastU, lastV, x0, y0;
    float current_angle, current_u, current_v;
    double delta_cum, prev_dist;
    bool latch;
    image_transport::Subscriber textureSub;
    Eigen::Matrix4d T, prevPose;


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
            // std::cout<<"tf retrieved, base to map is \n" <<pose<<std::endl;
        }

        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Received an exception trying to transform: %s", ex.what());
        }

        //pose=zeroTransform*pose;

        //std::cout<<"tf retrieved, ekf to base is 2 \n" <<pose<<std::endl;

        return pose;
    }

    Eigen::Matrix4d getLivoxtoBase(){


        Eigen::Matrix4d pose=Eigen::Matrix4d::Identity();

        try{


            tf::StampedTransform tempTf;
            listener1.waitForTransform("base_link", "livox_frame", ros::Time(0), ros::Duration(1.0) );

            listener1.lookupTransform("base_link", "livox_frame", ros::Time(0), tempTf);

            tf::Quaternion q=tempTf.getRotation();
            tf::Vector3 t=tempTf.getOrigin();
            pose(0,3)=t.x();
            pose(1,3)=t.y();
            pose(2,3)=t.z();

            Eigen::Quaterniond qqq(q.w(),q.x(),q.y(),q.z());
            Eigen::Matrix3d mmm(qqq);

            pose.block(0,0,3,3)=mmm;
        }

        catch (tf::TransformException& ex)
        {
            ROS_ERROR("Received an exception trying to transform: %s", ex.what());
        }


        return pose;
    }

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> voxelize(pcl::PointCloud<pcl::PointXYZ>::Ptr& temp_cloud, float leaf_size=0.01){

        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution_);
        octree.setInputCloud (temp_cloud);
        octree.addPointsFromInputCloud ();

        return octree;
    }

    void publish(cv::Mat& data, ros::Publisher& pub, pcl::PCLHeader header, std::string frame_id, double resolution, Eigen::Matrix4d& pose){

        std_msgs::Header h;
        pcl_conversions::fromPCL(header, h);

        sensor_msgs::ImagePtr msg;

        int dist=100; //grid size

        int width=data.size().width;
        int height=data.size().height;

        for(int i=0;i<height;i+=dist)
            cv::line(data,cv::Point(0,i),cv::Point(width,i),cv::Scalar(255,255,255));

        for(int i=0;i<width;i+=dist)
            cv::line(data,cv::Point(i,0),cv::Point(i,height),cv::Scalar(255,255,255));

        cv::Point A(100,100);   //modify this value to reposition the scale
        cv::Point B=A;
        B.y=B.y-5;
        B.x=B.x-10;
        cv::Point C(A.x+dist,A.y);

        std::ostringstream out;
        out.precision(1);
        out << std::fixed << resolution*(C.x-A.x);

        string res= out.str();   //std::to_string(resolution*100);
        string txt=" "+res+" m.";

        cv::putText(data, //target image
                    txt, //text "|1 m.|"
                    B, //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    0.85,
                    CV_RGB(255, 255, 255), //font color
                    2);

        cv::line(data, A, C, CV_RGB(255, 255, 255), 2.0, cv::LINE_8);

        cv::Point D=A;

        cv::Point E=C;

        D.y=D.y-20;
        E.y=E.y-20;

        cv::line(data, A, D, CV_RGB(255, 255, 255), 2.0, cv::LINE_8);
        cv::line(data, C, E, CV_RGB(255, 255, 255), 2.0, cv::LINE_8);

        cv::putText(data, //target image
                    "+", //text "~(o_o)~"
                    cv::Point(data.cols/2-15, data.rows-0), //top-left position
                    cv::FONT_HERSHEY_DUPLEX,
                    1.0,
                    CV_RGB(250, 250, 250), //font color
                    2);

        msg= cv_bridge::CvImage(std_msgs::Header(),"bgr8", data).toImageMsg();

        msg->header.stamp=h.stamp;
        msg->header.frame_id="base_link";

        live_texture_msgs::LiveTexture texture;
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
        // std::cout<<"texture point cloud size is "<<cloud->size()<<std::endl;

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);;
        pcl::copyPointCloud(*cloud_xy, *cloud2);
        //        std::cout<<cloud2->points[0].z<<std::endl;
        //        std::cout<<cloud2->points[10].z<<std::endl;
        //        std::cout<<cloud2->points[100].z<<std::endl;
        //        std::cout<<cloud2->points[1000].z<<std::endl;
        //        std::cout<<cloud2->points[0].x<<std::endl;
        //        std::cout<<cloud2->points[10].y<<std::endl;
        //        std::cout<<cloud2->points[100].x<<std::endl;
        //        std::cout<<cloud2->points[1000].y<<std::endl;
        float radius = 0.01;

        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree=voxelize(cloud2);
        
        //        Eigen::MatrixXi directions(1,3);

        //        directions << 1, 0 ,0;

        //sor.setSaveLeafLayout(true);
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

                // texture_temp=texture;
                bool green=false;

                pcl::PointXYZ searchPoint2;
                searchPoint2.x=searchPoint.x;
                searchPoint2.y=searchPoint.y;
                searchPoint2.z=0;



                //kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0
                std::vector<int> pointIdxVec;
                if (octree.radiusSearch (searchPoint2, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
                {

                    //std::cout<<"idx is "<<idx<<std::endl;
                    //std::cout<<"it happened"<<std::endl;
                    if (color=="red"  && mergeTextures_){

                        //std::cout << "red texture here "<<std::endl;
                        texture.at<cv::Vec3b>(v,u)[2]=255;
                        texture.at<cv::Vec3b>(v,u)[1]=0;
                        texture.at<cv::Vec3b>(v,u)[0]=0;

                        texture_temp.at<cv::Vec3b>(v,u)[2]=255;
                        texture_temp.at<cv::Vec3b>(v,u)[1]=0;
                        texture_temp.at<cv::Vec3b>(v,u)[0]=0;
                    }
                    if (color=="green" && mergeTextures_){
                        green=true;

                        int r=texture.at<cv::Vec3b>(v,u)[2];
                        int g=texture.at<cv::Vec3b>(v,u)[1];
                        int b=texture.at<cv::Vec3b>(v,u)[0];

                        if(r==0 && g==0 && b==0){

                            texture.at<cv::Vec3b>(v,u)[2]=0;
                            texture.at<cv::Vec3b>(v,u)[1]=255;
                            texture.at<cv::Vec3b>(v,u)[0]=0;
                        }
                        texture_temp.at<cv::Vec3b>(v,u)[2]=0;
                        texture_temp.at<cv::Vec3b>(v,u)[1]=255;
                        texture_temp.at<cv::Vec3b>(v,u)[0]=0;
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

                    int r= texture_temp.at<cv::Vec3b>(v,u)[2];
                    int g= texture_temp.at<cv::Vec3b>(v,u)[1];
                    int b= texture_temp.at<cv::Vec3b>(v,u)[0];
                    bool unboserved=true;

                    if ( (r==255 && g==0 && b==0) || (r==0 && g==255 && b==0) || (r==0 && g==0 && b==0) ){

                        unboserved=false;
                    }

                    if (mergeTextures_ &&  unboserved  && !keepBackground_  )
                    {
                        texture.at<cv::Vec3b>(v,u)[2]=0;   //r
                        texture.at<cv::Vec3b>(v,u)[1]=0;   //g
                        texture.at<cv::Vec3b>(v,u)[0]=0;  //b
                    }
                }
            }
        }

        //erode(texture,1);
        //dilate(texture,1);

        texture_temp2=texture.clone();
        if (pubSwtich){
            publish(texture, pub, cloud->header ,cloud->header.frame_id, resolution_, newPose);
        }
        //std::cout<<texture.size()<<std::endl;
        //cv::imshow( "Undistorted", texture );
        //cv::waitKey(0);

    }


    void mapCb(const sensor_msgs::PointCloud2ConstPtr& inliers, const sensor_msgs::PointCloud2ConstPtr& outliers)
    {

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

        pcl::transformPointCloud (*inliers_pcl, *inliers_transformed, newPose);
        pcl::transformPointCloud (*outliers_pcl, *outliers_transformed, newPose);

        *inliers_pcl_acc=*inliers_transformed+*inliers_pcl_acc;
        *outliers_pcl_acc=*outliers_transformed+*outliers_pcl_acc;

        *inliers_pcl_acc_tmp=*inliers_pcl_acc;
        *outliers_pcl_acc_tmp=*outliers_pcl_acc;

        //*inliers_pcl_acc=*inliers_transformed;
        //*outliers_pcl_acc=*outliers_transformed;

        //*inliers_pcl_acc=*inliers_pcl;
        //*outliers_pcl_acc=*outliers_pcl;

        init=true;

        tempfn(true, newPose);

    }

    void tempfn(bool newData, Eigen::Matrix4d newPose){
        Eigen::Matrix4d pose;
        Eigen::Matrix4d pose_inv;
        Eigen::Matrix2d temp;


        if (init){
            pose=getPoseTf();
            pose_inv=pose.inverse();
            temp=pose_inv.block(0,0,2,2);
        }

        if (init && newData){ //stop and map!
            //Eigen::Matrix4d pose=getPoseTf();
            Eigen::Matrix4d pose_inv=newPose.inverse();

            pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl_acc_tmp2 (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl_acc_tmp2 (new pcl::PointCloud<pcl::PointXYZ>);

            std::cout<<"inliers size is "<<inliers_pcl_acc->size()<<std::endl;
            pcl::transformPointCloud (*inliers_pcl_acc, *inliers_pcl_acc_tmp2, pose_inv);
            pcl::transformPointCloud (*outliers_pcl_acc, *outliers_pcl_acc_tmp2, pose_inv);
            std::cout<<"new pose is "<<newPose<<std::endl;
            //cv::Mat texture2=texture.clone();
            //cv::Mat texture2(texture.size(),texture.type());
            // texture.copyTo(texture2);
            if (inliers_pcl_acc->size()>0){
                publish_colored_texture(texture,newPose,inliers_pcl_acc_tmp2, mergedTexturePub,"green", false); //todo: replace pose by the stampedPose (newpose)
            }

            if (outliers_pcl_acc->size()>0){

                publish_colored_texture(texture,newPose,outliers_pcl_acc_tmp2, mergedTexturePub,"red", true);
            }

            //           Eigen::Rotation2D<double> pose_inv_rot(temp);
            //            lastYaw = pose_inv_rot.angle();
            lastYaw=current_angle;
            //            float du=(pose_inv(1,3)/resolution_);
            //            float dv=(pose_inv(0,3)/resolution_);
            //            lastU=du;
            //            lastV=dv;
            lastU=current_u;
            lastV=current_v;
            delta_cum=0;
            x0=0; y0=0;
            latch=false;
            prevPose=T;
        }

        if (init && !newData){  //dynamic mapping

if (low_level_layer_){

		

            pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl_acc_tmp3 (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl_acc_tmp3 (new pcl::PointCloud<pcl::PointXYZ>);

            std::cout<<"inliers size is "<<inliers_pcl_acc->size()<<std::endl;

            pcl::transformPointCloud (*inliers_pcl_acc_tmp, *inliers_pcl_acc_tmp3, pose.inverse().eval());
            pcl::transformPointCloud (*outliers_pcl_acc_tmp, *outliers_pcl_acc_tmp3, pose.inverse().eval());

            //std::cout<<"new pose is "<<newPose<<std::endl;
            //cv::Mat texture2=texture.clone();
            //cv::Mat texture2(texture.size(),texture.type());
            // texture.copyTo(texture2);

            if (inliers_pcl_acc_tmp3->size()>0){
                publish_colored_texture(texture_scan,newPose,inliers_pcl_acc_tmp3, mergedTexturePub,"green", false); //todo: replace pose by the stampedPose (newpose)
            }

            if (outliers_pcl_acc_tmp3->size()>0){

                publish_colored_texture(texture_scan,newPose,outliers_pcl_acc_tmp3, mergedTexturePub,"red", true);
            }




}
else{




            Eigen::Rotation2D<double> pose_inv_rot(temp);
            // double yaw = pose_inv_rot.angle()-lastYaw;
            double yaw = current_angle - lastYaw;

            Eigen::Matrix4d cPose=prevPose.inverse()*T;
            cPose=cPose.inverse().eval();

            //            if ( (x0==0 && y0==0 && delta_cum <0) || (latch)){
            //                latch=true;
            //                x0 = -delta_cum*cos(yaw) + x0;
            //                y0 = -delta_cum*sin(yaw) + y0;

            //            }
            //            else
            //            {

            //                x0 = delta_cum*cos(yaw) + x0;
            //                y0 = delta_cum*sin(yaw) + y0;

            //            }


            //            double x=x0;
            //            double y=y0;
            //            delta_cum=0;

            Eigen::Matrix2d temp2= cPose.block(0,0,2,2);
            Eigen::Rotation2D<double> pose2D(temp2);
            yaw=pose2D.angle();
            double x=cPose(0,3);
            double y=cPose(1,3);


            //           float du=(pose_inv(1,3)/resolution_)-lastU;
            //           float dv=(pose_inv(0,3)/resolution_)-lastV;

            float du=current_u-lastU;
            float dv=current_v-lastV;
            // std::cout<<"yaw rad is "<<yaw<<" x is "<<x<<" y is "<<y<<std::endl;
            float warp_values[] = { 1.0, 0.0, -y/resolution_, 0.0, 1.0, -x/resolution_ };

            cv::Mat translation_matrix = cv::Mat(2, 3, CV_32F, warp_values);
            double angle = yaw*180/PI;


            double scale = 1;
            cv::Mat warp_dst = cv::Mat::zeros( texture_temp2.rows, texture_temp2.cols, texture_temp2.type() );
            cv::Point center = cv::Point( warp_dst.cols/2, warp_dst.rows/1-20 );
            cv::Mat rot_mat = cv::getRotationMatrix2D( center, angle, scale );
            //std::cout<<"rot_mat is "<<rot_mat<<std::endl;

            cv::warpAffine( texture_temp2, warp_dst, rot_mat, warp_dst.size() );

            cv::warpAffine(warp_dst, warp_dst, translation_matrix, warp_dst.size());

            publish(warp_dst, mergedTexturePub, inliers_pcl_acc->header ,inliers_pcl_acc->header.frame_id, resolution_, pose);



}



















        }



    }

    void scan_callback(const sensor_msgs::PointCloud2ConstPtr& inliers, const sensor_msgs::PointCloud2ConstPtr& outliers){

        if (init){

            pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_pcl (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg( *inliers, *inliers_pcl);

            pcl::PointCloud<pcl::PointXYZ>::Ptr outliers_pcl (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg( *outliers, *outliers_pcl);


            Eigen::Matrix4d newPose=getPoseTf()*getLivoxtoBase();

            pcl::transformPointCloud (*inliers_pcl, *inliers_pcl, newPose);
            pcl::transformPointCloud (*outliers_pcl, *outliers_pcl, newPose);

            *inliers_pcl_acc_tmp=*inliers_pcl+*inliers_pcl_acc_tmp;
            *outliers_pcl_acc_tmp=*outliers_pcl+*outliers_pcl_acc_tmp;
        }




    }

    void odom_callback(const nav_msgs::OdometryConstPtr& odom){


        double tx=odom->pose.pose.position.x;
        double ty=odom->pose.pose.position.y;
        double tz=odom->pose.pose.position.z;

        double qx=odom->pose.pose.orientation.x;
        double qy=odom->pose.pose.orientation.y;
        double qz=odom->pose.pose.orientation.z;
        double qw=odom->pose.pose.orientation.w;

        Eigen::Quaterniond q(qw,qx,qy,qz);
        Eigen::Matrix3d M(q);
        Eigen::Matrix2d temp=M.block(0,0,2,2);
        Eigen::Rotation2D<double> rot(temp);

        current_angle=-rot.angle();

        double c_dist=sqrt(tx*tx+ty*ty);
        if (prev_dist==-1){
            prev_dist=c_dist;
        }
        double delta=(-prev_dist+c_dist);

        delta_cum=delta+delta_cum;
        prev_dist=c_dist;



        current_u=ty/resolution_;
        current_v=tx/resolution_;

        T=Eigen::Matrix4d::Identity();


        T(0,3)=tx;
        T(1,3)=ty;
        T.block(0,0,3,3)=M;
        T=getPoseTf();
        //Eigen::Matrix4d pose;

        //          Eigen::Matrix4d pose_inv;
        //          Eigen::Matrix2d temp;

        //          pose_inv=pose.inverse();
        //              temp=pose_inv.block(0,0,2,2);
        //          Eigen::Matrix4d T=Eigen::Matrix4d::Identity();
        //          T.block(0,0,3,3)=M;


    }


    void img_callback(const sensor_msgs::ImageConstPtr & img_msg)  //ROS callback
    {
        try
        {
            cv::Mat img(cv_bridge::toCvCopy(img_msg,"bgr8")->image);
            texture_scan=img;
        }

        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;

        }

        if (getTexture){
            std::cout<<"trigger ON..."<<std::endl;
            getTexture=false;
            try
            {
                cv::Mat img(cv_bridge::toCvCopy(img_msg,"bgr8")->image);
                texture=img;
                texture_temp=img.clone();

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
        else {
            tempfn(false, Eigen::Matrix4d::Identity());
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
        n.param<bool>("low_level_layer", low_level_layer_, true);
        n.param<bool>("merge_textures", mergeTextures_, false);
        n.param<bool>("keep_merged_texture_background", keepBackground_, false);
        n.param<double>("rotation_tuning", rotationTuning, -4);
        fx=608.151123046875; fy=606.9262084960938;
        cx=328.7842102050781; cy=242.90274047851562;
        intrinsics=Eigen::Matrix3f::Identity();
        intrinsics(0,0)=fx; intrinsics(0,2)=cx;
        intrinsics(1,1)=fy; intrinsics(1,2)=cy;
        intrinsics(2,2)=1;
        prev_dist=-1;
        getTexture=false;

        inliers_pcl_acc = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        outliers_pcl_acc = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        inliers_pcl_acc_tmp = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        outliers_pcl_acc_tmp = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        init=false;

        inliersSub.subscribe(n, "inliers", 1);
        outliersSub.subscribe(n, "outliers", 1);

        inliersSubTmp.subscribe(n, "inlier_scan", 1);
        outliersSubTmp.subscribe(n, "outlier_scan", 1);

        //coloredScanPublisher = n.advertise<pcl::PointCloud<pcl::PointXYZRGBAI> > ("livox/lidar_colored", 1);

        inliersTexturePub = n.advertise<live_texture_msgs::LiveTexture> ("texture_inliers", 1);
        outliersTexturePub = n.advertise<live_texture_msgs::LiveTexture> ("texture_outliers", 1);




        if (mergeTextures_){

            mergedTexturePub = n.advertise<live_texture_msgs::LiveTexture> ("texture_merged", 1);

        }
        // scan_sub_ = nh_.subscribe("inliers",1,&Mapper::inliers_callback,this);

        //encoder_sub_ = n.subscribe("encoder_sub",1,&Mapper::rolling_callback,this);

        odomSub=n.subscribe("/odom",1,&Mapper::odom_callback,this);
        textureSub=it.subscribe("/texture_in",1,&Mapper::img_callback,this,transport);
        // inliers_texture = it.advertise("obs_img", 1);

        sync.reset(new Sync(MySyncPolicy(10), inliersSub, outliersSub));
        syncTmp.reset(new SyncTmp(MySyncPolicyTmp(10), inliersSubTmp, outliersSubTmp));

        sync->registerCallback(boost::bind(&Mapper::mapCb, this, _1, _2));
        syncTmp->registerCallback(boost::bind(&Mapper::scan_callback, this, _1, _2));
    }
};

int main(int argc, char * argv[]){


    ros::init(argc, argv, "crawler_texture_layers");
    Mapper var;


    ros::spin();

}
