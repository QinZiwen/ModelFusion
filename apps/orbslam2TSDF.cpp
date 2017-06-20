#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>

#include <kfusion/kinfu.hpp>

#include "orbslam2/System.h"

using namespace kfusion;
using namespace std;

struct KinFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;
    }

    KinFuApp(const string &strVocFile, const string &strSettingsFile) : exit_ (false),  iteractive_mode_(false), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );
		orbslam2_ = new ORB_SLAM2::System(strVocFile, strSettingsFile, ORB_SLAM2::System::RGBD, true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }
    
    ~KinFuApp()
	{
		if(orbslam2_)
		{
			orbslam2_->Shutdown();
			delete orbslam2_;
		}
	}

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 3;
        if (iteractive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    bool execute(vector<double> &vTimestamps, vector<string> &vstrRGB, vector<string> &vstrDepth)
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
		double tframe = 0;

        for (int i = 0; !exit_ && !viz.wasStopped() && i < vstrRGB.size(); ++i)
        {
			tframe = vTimestamps[i];
            image = cv::imread(vstrRGB[i], CV_LOAD_IMAGE_UNCHANGED);
			depth = cv::imread(vstrDepth[i], CV_LOAD_IMAGE_UNCHANGED);
            if (image.empty() || depth.empty())
                return std::cout << "Read image fail!" << std::endl, false;
			
            //cv::imshow("Image", image);
			//cv::waitKey();
			
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

			cv::Mat T;
            {
                SampledScopeTime fps(time_ms); (void)fps;
                //has_image = kinfu(depth_device_);
				T = orbslam2_->TrackRGBD(image,depth,tframe);
            }
                        				
			cout << "T = " << endl << T << endl;
			if(T.empty())
			{				
				continue;
			}
			
			cv::Mat Rcw = T.rowRange(0,3).colRange(0,3);
			cv::Mat tcw = T.rowRange(0,3).col(3);

			cv::Mat twc = -Rcw.t()*tcw;
			
			Affine3f pose(Rcw.t(), twc);
			has_image = kinfu.integrateDpethUsePose(depth_device_, pose);

            show_raycasted(kinfu);

            show_depth(depth);

            if (!iteractive_mode_)
                viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch(key)
            {
				case 't': case 'T' : take_cloud(kinfu); break;
				case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
				case 27: exit_ = true; break;
				case 32: pause_ = !pause_; break;
            }

            viz.spinOnce(3, true);
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    KinFu::Ptr kinfu_;
	ORB_SLAM2::System *orbslam2_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void loadImages(const string databasePath, const string associatedFile, vector<double> &vTimestamps, vector<string> &vstrRGB, vector<string> &vstrDepth)
{
	ifstream fAssociation(associatedFile);
	if(!fAssociation.is_open())
	{
		cout << associatedFile << " dose not exist!" << endl;
		exit(0);
	}
	
	while(!fAssociation.eof())
	{
		string s;
		getline(fAssociation, s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			
			double t;
			string sRGB, sD;
			
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrRGB.push_back(databasePath + "/" + sRGB);
			ss >> t;
			ss >> sD;
			vstrDepth.push_back(databasePath + "/" + sD);
		}
	}
	
	cout << "Total number of images = " << vstrRGB.size() << endl;
}

int main (int argc, char* argv[])
{
	if(argc != 5)
	{
		cout << "Usage: ./offLine path_to_vocabulary path_to_settings database_path image_associated_file" << endl;
		return -1;
	}
	
	string vocabularyPath(argv[1]);
	string settingPath(argv[2]);
	string databasePath(argv[3]);
	string associatedFile(argv[4]);
	
	vector<double> vTimestamps;
	vector<string> vstrRGB;
	vector<string> vstrDepth;
	
	cout << "databasePath = " << databasePath << endl;
	cout << "associatedFile = " << associatedFile << endl;
	loadImages(databasePath, associatedFile, vTimestamps, vstrRGB, vstrDepth);
	
    KinFuApp app(vocabularyPath, settingPath);

    // executing
    try { app.execute (vTimestamps, vstrRGB, vstrDepth); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
