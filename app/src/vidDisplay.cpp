/**
 * The above code is a C++ program that reads a video, extracts its metadata, and displays the video
 * while allowing the user to apply different filters by pressing different keys.
 * 
 * @return The main function is returning an integer value of 0.
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <filter.h>


using namespace std;
using namespace cv;

// The purpose of this class is to read a video, extract its relevent metadata,
//     and display the video. While displaying the video, the user can apply different filters,
//     by tapping appropriate keys.

// Member variables:
//     - inputVideo : A VideoCapture object which is used for extracting frames.
//     - readStat : A boolean Variable that stores 'true' if the video was read successfully,
//              else it stores 'false'.
//     - vidFrameCount : A double that stores the number of frames in the video.
//     - vidDuration : A double that stores the duration of th video in seconds.
//     - vidDims : that stores the resolution of the video in a cv::Size object.
// Member functions:
//     - displayVideo() :         // Display video function that displays video in interactive mode by allowing
        //  the user to click different keys to activate different filters.

class VideoOps
{   
    public:
        VideoCapture *inputVideo {};
        bool readStat {};
        double vidFrameCount {};
        double vidFrameRate {};
        double vidDuration {};
        double vidDims[2] {};
    
    
    public:
        // Constructor to initialize the different member variables.
        VideoOps(string vidPath = "0")
        {
            if(vidPath == "0")
                inputVideo = new VideoCapture(0);
            else
                inputVideo = new VideoCapture(vidPath, cv::CAP_FFMPEG);   

            readStat = inputVideo->isOpened();
            
            vidFrameCount = inputVideo->get(CAP_PROP_FRAME_COUNT);

            vidFrameRate = inputVideo->get(CAP_PROP_FPS);

            vidDuration = static_cast<double>(vidFrameCount) / vidFrameRate;

            vidDims[0] = inputVideo->get(CAP_PROP_FRAME_WIDTH);
            vidDims[1] = inputVideo->get(CAP_PROP_FRAME_HEIGHT);
        }
 
        //Displays the video in an interactive mode, where the user can enter 
        //  different keys to toggle different filters. The keys along with the filters they 
        //  activate are as shown below.
        //  - 's' - Save current frame as a .jpg file.
        //  - 'q' - Stop the video stream and exit the program.
        //  - 'g' - Apply OpenCV's default colour to grayscale function.
        //  - 'h' - Apply Custom colour to grayscale function.
        //  - '1' - Apply Sepia tone filter.
        //  - '2' - Increase contrast using 'normalized squaring' method. (extension 2).
        //  - '3' - Increase contrast using 'shifted and scaled sigmoid' method. (extension 3).
        //  - '4' - Apply Gaussian filter naively (element wise operations) (inefficent).
        //  - '5' - Apply Gaussian filter relatively efficiently (slice operations).
        //  - 'x' - Apply Sobel X seperated filters.
        //  - 'y' - Apply Sobel Y seperated filters.
        //  - 'm' - Apply Magnitude function, to display Magnitude of gradiant using the Sobel
        //              filters.
        //  - 'l' - Apply Blurring and Quantize the channels into 10 bins.
        //  - 'f' - Detect faces in the video stream.
        //  - '7' - Detect faces in the video stream and blurr the background.
        //  - '8' - Detect faces in the video stream and convert the background to grayscale.
        //  - '9' - Make negative.
        //  - '0' - Cartoonize the video feed. (Extension 1).
        void displayVideo()
        {
            Mat frame;
            string outpath_prefix {"../bin/frame_"}, outpath_suffix{".jpg"};
            /* The line `int frameno{1};` is declaring and initializing an integer variable named
            `frameno` with a value of 1. This variable is used to keep track of the frame number
            when saving frames as .jpg files. */
            int frameno{1};

            char key = ' ';
            char oldkey = ' '; //for exclusively 2 step sequential sequential operations 
            bool ss = false;

            int count{1};
            namedWindow("Video", 1); 
            for(;;) {
                *inputVideo >> frame; 
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                

                int temp = waitKey(10);

                if (temp != -1)
                    key = temp;

                temp = key;
                // cout << temp << " " << key  << "\n";

                // to save or not to save
                if(key == 's') {
                    ss = true;
                    key = oldkey;  // changing from "toggle" to "tapper";
                }

                //########################### effects
                if( key == 'q') {
                    cout << key;
                    break;
                }
                
                // Task 3.
                else if(key == 'g'){
                    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                }

                // Task 4.
                else if(key == 'h'){
                    betterBGRtoBW(frame, frame);
                }

                // Task 5.
                else if(key == '1'){
                    Sepia(frame, frame);
                }

                //Extension 2 (Contrast method 1)
                else if(key == '2'){
                    ContrastSquare(frame, frame);
                }

                //Extension 3 (Contrast method 2)
                else if(key == '3'){
                    ContrastSigmoid(frame, frame);
                }
                
                // Task 6.1. 
                else if(key == '4'){
                    blur5x5_1(frame, frame);
                }

                // Task 6.2.
                else if(key == '5'){
                    blur5x5_2(frame, frame);
                }

                // Task 7.1.
                else if(key == 'x'){
                    sobelX3x3(frame, frame);
                    vizderivative(frame);
                }

                // Task 7.2.
                else if(key == 'y'){
                    sobelY3x3(frame, frame);
                    vizderivative(frame);
                }
                
                // Task 8.
                else if(key == 'm'){
                    Mat dx = frame.clone();
                    Mat dy = frame.clone();
                    sobelX3x3(dx, dx);
                    sobelY3x3(dy, dy);
                    magnitude(dx, dy, frame);
                    vizderivative(frame);
                }
                
                // Task 9.
                else if(key == 'l'){
                    blurQuantize(frame, frame, 10);
                }   

                // Task 10.
                else if(key == 'f'){
                    findFaces(frame);
                }    

                // Task 11.1.
                else if(key == '7'){
                    envblurr(frame);
                }    

                // Task 11.2.
                else if(key == '8'){
                    bwenv(frame);
                }  

                // Task 11.3.
                else if(key == '9'){
                    makeNegative(frame);
                }   

                //Extension 3. Cartoonization
                else if(key == '0'){
                    makecartoon(frame);
                }   
                
                //######################## effects end

                if (ss == true)
                {
                    imwrite(outpath_prefix + to_string(frameno) + outpath_suffix, frame);
                    frameno++;
                    ss = false;
                }

                oldkey = key;
                imshow("Video", frame);
                
            };

        }   

};

//Driver program
int main()
{
    //Link to the http webcam stream.
    string source = "http://50.010.24.174:5000/video_feed"; //needs to be changed according to your URL. 
    VideoOps vid{source};
    
    vid.displayVideo();
    
    return 0;
}