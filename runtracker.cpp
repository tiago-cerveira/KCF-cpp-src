#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include <boost/algorithm/string.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "cnpy.h"

#include <dirent.h>

using namespace std;
using namespace cv;
using namespace chrono;

vector<Mat> nextFrame(cnpy::npz_t data, int whichFrame){
    int i, j, k, ii, jj;
    float f;
    char *p;
    string img_name = "arr_" + to_string(whichFrame);
    //cout << img_name << endl;
    cnpy::NpyArray img = data[img_name];
    vector<unsigned int> shape = img.shape;
    //cout <<"("<<shape[0]<<","<<shape[1]<<","<<shape[2]<<")"<<endl;

    //int ***datanew = new int **[shape[0]];
    vector<Mat> cv_imgs;
    for(jj = 0; jj < shape[2]; jj++){
        cv_imgs.push_back(Mat(shape[0], shape[1], CV_8UC3, double(0)));
    }

    p = (char*)&f;
    for(i = 0; i < shape[0]; i++){
        //datanew[i] = new int *[shape[1]];
        for(j = 0; j < shape[1]; j++){
            //datanew[i][j] = new int[shape[2]];
            for(k = 0; k < shape[2]; k++){
//                for(ii = 0; ii < 4; ii++){
//                    *(p+ii) = img.data[i*shape[1]*shape[2]*4+j*shape[2]*4+k*4+ii];
//                }
                //cout <<"("<<i<<","<<j<<","<<k<<")\t";
                //datanew[i][j][k] = img.data[i*shape[1]*shape[2]+j*shape[2]+k];
                //cout << "f: " << f << "\t";
//                f *= 50;
//                cv_img.at<Vec3b>(i, j)[k] = f;
                cv_imgs[k].at<Vec3b>(i, j) = img.data[i*shape[1]*shape[2]+j*shape[2]+k];
                //cout << float(cv_imgs[k].at<Vec3b>(i, j)[0]) << endl;
                //sleep(1);
            }
        }
    }
    //imshow("Image", cv_imgs[0]);
    //waitKey();
    return cv_imgs;
}

int main(int argc, char* argv[]){

    if (argc > 5) return -1;

    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = false;

    for(int i = 0; i < argc; i++){
        if ( strcmp (argv[i], "hog") == 0 )
            HOG = true;
        if ( strcmp (argv[i], "fixed_window") == 0 )
            FIXEDWINDOW = true;
        if ( strcmp (argv[i], "singlescale") == 0 )
            MULTISCALE = false;
        if ( strcmp (argv[i], "show") == 0 )
            SILENT = false;
        if ( strcmp (argv[i], "lab") == 0 ){
            LAB = true;
            HOG = true;
        }
        if ( strcmp (argv[i], "gray") == 0 )
            HOG = false;
        cout << argv[i];
        cout << " ";
    }
    cout << endl;

    // Create KCFTracker object
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // Frame readed
    vector<Mat> frames;
    Mat frame;

    // Tracker results
    Rect result;

    // Read groundtruth for the 1st frame
    ifstream groundtruthFile;
    string groundtruth;

    groundtruth = "compressed_seq/" + string(argv[1]) + "_groundtruth.txt";

    groundtruthFile.open(groundtruth);
    string firstLine;

    getline(groundtruthFile, firstLine);
    groundtruthFile.close();

    istringstream ss(firstLine);

    // Read groundtruth like dumb
    float x1, y1, x2, y2, x3, y3, x4, y4;
    char ch;
    ss >> x1;
    ss >> ch;
    ss >> y1;
    ss >> ch;
    ss >> x2;
    ss >> ch;
    ss >> y2;
    ss >> ch;
    ss >> x3;
    ss >> ch;
    ss >> y3;
    ss >> ch;
    ss >> x4;
    ss >> ch;
    ss >> y4;

//    cout << "x1 " << x1 << endl;
//    cout << "y1 " << y1 << endl;
//    cout << "x2 " << x2 << endl;
//    cout << "y2 " << y2 << endl;
//    cout << "x3 " << x3 << endl;
//    cout << "y3 " << y3 << endl;
//    cout << "x4 " << x4 << endl;
//    cout << "y4 " << y4 << endl;

    // Using min and max of X and Y for groundtruth rectangle
    float xMin =  min(x1, min(x2, min(x3, x4)));
    float yMin =  min(y1, min(y2, min(y3, y4)));
    float width = max(x1, max(x2, max(x3, x4))) - xMin;
    float height = max(y1, max(y2, max(y3, y4))) - yMin;

    cout << "xMin   " << xMin << endl;
    cout << "yMin   " << yMin << endl;
    cout << "width  " << width << endl;
    cout << "height " << height << endl;

    // Read Images
    string sequence_name = string("compressed_seq/") + argv[1] + ".npz";
    //cout << sequence_name << endl;
    cnpy::npz_t my_npz = cnpy::npz_load(sequence_name);
    int seq_size = my_npz.size();

    ifstream listFramesFile;
    vector<string> strs;
    boost::split(strs, argv[1], boost::is_any_of("_"));
    //cout << strs[0] << endl;
    string listFrames = "data_seq/" + strs[0] + "/img/images.txt";
    listFramesFile.open(listFrames);
    string frameName;


    // Write Results
    ofstream resultsFile;
    string resultsPath = "output.txt";
    resultsFile.open(resultsPath);

    Rect raw_res;

    // Frame counter
    int nFrames = 0;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    while(nFrames < seq_size){

        // Read each frame
        frames = nextFrame(my_npz, nFrames);
        getline(listFramesFile, frameName);
        //cout << frameName << endl;
        frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
        //cout << "cols: " << frame.cols << endl;
        //cout << "rows: " << frame.rows << endl;


        // First frame, give the groundtruth to the tracker
        if (nFrames == 0) {
            tracker.init( Rect(xMin, yMin, width, height), frames);
            //rectangle( frames[1], Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar(0, 255, 255), 1, 8);
            resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
        }
        // Update
        else{
            result = tracker.update(frames);
            raw_res.x = (result.x * frame.cols)/frames[0].cols;
            raw_res.y = (result.y * frame.rows)/frames[0].rows;
            raw_res.width = (result.width * frame.cols)/frames[0].cols;
            raw_res.height = (result.height * frame.rows)/frames[0].rows;
            //cout << raw_res.x <<","<< raw_res.y <<","<< raw_res.width <<","<< raw_res.height << endl;
            rectangle(frame, Point( raw_res.x, raw_res.y ), Point( raw_res.x+raw_res.width, raw_res.y+raw_res.height), Scalar(0, 255, 255), 1, 8);
            //rectangle(frames[0], Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar(0, 255, 255), 1, 8);

            resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
        }

        nFrames++;

        if (!SILENT){
            namedWindow("Raw Image", WINDOW_NORMAL);
            resizeWindow("Raw Image", frame.cols * 1.5f, frame.rows * 1.5f);
            imshow("Raw Image", frame);

//            namedWindow("CNN Image", WINDOW_NORMAL);
//            resizeWindow("CNN Image", 600, 600);
//            imshow("CNN Image", frames[0]);

            waitKey(1);
        }
    }
    resultsFile.close();

    //listFile.close();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    cout << "tracking took " << setprecision(2) << fixed << time_span.count() << " seconds" << endl;
    cout << "average FPS: "<< setprecision(2) << fixed << nFrames/time_span.count() << " [FPS]" << endl;
    cout << "total frames: " << nFrames << endl;

}
