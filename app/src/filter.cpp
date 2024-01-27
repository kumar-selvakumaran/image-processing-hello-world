#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <iostream>
#include <filter.h>
#include <faceDetect.h>

using namespace std;
using namespace cv;

// This function is used to convert a cv::Mat object from BGR colour space to grayscale. This is done by 
//    setting the pixel-wise max channel value as the value for all channels.
   
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
    
int greyscale( Mat &src, Mat &dst )
{
    src.copyTo(dst);
    Mat channels[3];
    split(dst, channels);
    Mat selCh;
    max(channels[0], channels[1], selCh);
    max(selCh, channels[2], selCh);
    channels[0] = selCh;
    channels[1] = selCh;
    channels[2] = selCh;
    merge(channels, 3, dst);
    return 0;
}

// This function is used to apply a Sepia ton filter a image/frame. It takes input as acv::Mat object.
   
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame..  
void Sepia(cv::Mat &src, cv::Mat &dst)
{
    // 0.393, 0.349, 0.272,    // Red coefficients
    // 0.769, 0.686, 0.534,    // Green coefficients
    // 0.189, 0.168, 0.131     // Blue coefficients
    src.copyTo(dst);
    Mat channelsOg[3], channelsProc[3];
    split(src, channelsOg);
    split(dst, channelsProc);
    channelsProc[2] = (0.393 * channelsOg[2]) + (0.769* channelsOg[1]) + (0.769 * channelsOg[0]);
    channelsProc[1] = (0.349 * channelsOg[2]) + (0.686 * channelsOg[1]) + (0.168 * channelsOg[0]);
    channelsProc[0] = (0.272 * channelsOg[2]) + (0.534 * channelsOg[1]) + (0.131 * channelsOg[0]);
    merge(channelsProc, 3, dst);
}

// This function is used to print a slice of a cv:Mat object for debugging purposes. It takes input as a 
//    cv::Mat object.
   
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.  
void printmat(Mat* src)
{
    Mat interim = *src;
    Mat vizslice(interim(Range(140,145) , Range(140,145)));
    cout << "\nmatrix chunk : \n" << format(vizslice, Formatter::FMT_NUMPY ) <<"\n";  
}

//   EXTENSION 2
//   This function is used to increase the contrast of the input image/frame. The following operations are applied
//     on the input frame as shown below:
//     input frame/image -> Min-Max normalization -> element wise squaring -> inverse normalization (using original 
//         Minimum and Maximum values).

//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame. 
void ContrastSquare(Mat &src, Mat &dst )
{
    src.copyTo(dst);
        
    double minVal;
    double maxVal;

    dst.convertTo(dst, CV_64F);
    
    minMaxLoc(dst, &minVal, &maxVal);
    dst = (dst - minVal) / (maxVal - minVal);
    pow(dst,  2, dst);
    dst = (dst * (maxVal - minVal)) + minVal;
    dst.convertTo(dst, CV_8U);
}

// EXTENSION 2
// This function is used to increase the contrast of the input image/frame. The following operations are applied
//     on the input frame as shown below:
//     input frame/image -> Min-Max normalization -> y = 1/(1+(60*(e^(-9x)))) ->  elementwise scalar multiplication 
//       by 255.
//The operations are computed channel by channel.
//
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
void ContrastSigmoid(Mat &src, Mat &dst )
{
    src.copyTo(dst);
    dst.convertTo(dst, CV_32F);
    double minVal;
    double maxVal;
    _InputArray procInterim(dst);
    minMaxLoc(procInterim, &minVal, &maxVal);
    dst = (dst - minVal) / (maxVal - minVal);
    exp((-9*dst), dst);
    dst = 1 / (1 + (60*dst));
    dst *= 255;
    dst.convertTo(dst, CV_8U);
}


// This function is used to implement custom colour grayscale conversion. The following operations are applied
//     on the input frame as shown below:
//     input frame/image -> Increase contrast using "normalized squaring method" -> Converting to grayscale by setting each
//         pixel value to the maximum channel value of that pixel.

//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame. 
void betterBGRtoBW(Mat &src, Mat &dst )
{
    src.copyTo(dst);
    ContrastSquare(dst, dst);
    greyscale(dst, dst);
}

// This function is used to implement Gaussian blurring on the image/frame using the below filter.
//     [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1]
// The blurring is done by carrying out the correlation operation using the above filter. The sum of products is found for
//     each window of the Gaussian kernel's size in an element wise fashion.
// The operations are channel by channel.

//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
int blur5x5_1(Mat &src, Mat &dst )
{
    Mat procFrame = src.clone();

    float data[25] = { 1, 2, 4, 2, 1, 2, 4, 8, 4, 2, 4, 8, 16, 8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1} ;
    Mat gblurr = Mat(5, 5, CV_32F, data);

    Size dimsFrame = procFrame.size();
    int numColsFrame = dimsFrame.width;
    int numRowsFrame = dimsFrame.height;
    
    Size dimsBlurr = gblurr.size();
    int numColsBlurr =  dimsBlurr.width;
    int numRowsBlurr = dimsBlurr.height;
    int sumBlurr = sum(gblurr)[0];

    int r, c, ch; 
    int offsetX, offsetY;

    Mat procChannels[3];
    split(procFrame, procChannels);

    int numiters = 0;

    for (ch = 0; ch < 3; ch ++)
    {
        for (offsetX = 0 ; offsetX < (numRowsFrame - numRowsBlurr + 1) ; offsetX++)
        {
            for (offsetY = 0 ; offsetY < (numColsFrame - numColsBlurr + 1); offsetY++)
            { 
                double blurrVal = 0;
                for(r = 0; r < numRowsBlurr; r++)
                {
                    for(c = 0; c < numColsBlurr; c++)
                    {
                        blurrVal = blurrVal + (gblurr.at<float>(r,c) * static_cast<int>(procChannels[ch].at<uchar>(offsetX + r, offsetY + c)));
                        numiters++;
                    }
                }
                procChannels[ch].at<uchar>(offsetX+2, offsetY+2) = static_cast<uchar>(blurrVal/sumBlurr);
            }
        }
    }
    
    merge(procChannels, 3, procFrame);
    
    dst = procFrame; 

    return 1;
}

// This function is used to implement Gaussian blurring on the image/frame using the below filter.
//     [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1]
// The blurring is done by carrying out the correlation operation using the above filter. The filter is split into its
//     seperated versions, and is each of the seperated filters is applied column wise and row-wise respectively (not
//     element-wise). More efficent than  blur5x5_1().
// The operations are computed channel by channel.

//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
int blur5x5_2(Mat &src, Mat &dst )
{
    Mat procFrame = src.clone();

    float filter2d[25] = { 1, 2, 4, 2, 1, 2, 4, 8, 4, 2, 4, 8, 16, 8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1} ;
    float filter1d[5] = {1, 2, 4, 2, 1};
    
    Mat gblurrCh = Mat(5, 5, CV_32F, filter2d);
    Mat gblurr1dCh = Mat(1, 5, CV_32F, filter1d);
    
    Size dimsFrame = procFrame.size();
    int numColsFrame = dimsFrame.width;
    int numRowsFrame = dimsFrame.height;

    Size dimsFilter = gblurr1dCh.size();
    int numColsFilter = dimsFilter.width;
    int numRowsFilter = dimsFilter.height;
    
    int sumBlurr = sum(gblurr1dCh)[0];
    
    int r, c, ch; 

    Mat ogChannels[3], procChannels[3];

    procFrame.convertTo(procFrame, CV_32FC3);

    split(procFrame, procChannels);

    split(procFrame, ogChannels);
    
    int numiters = 0;

    int offset = numColsFilter / 2;

    for(ch = 0; ch < 3; ch++)
    {
        for(c = 0; c < (numColsFrame - numColsFilter + 1); c++)
        {
            Mat colSlice = ogChannels[ch].colRange(c, c+numColsFilter); 
            Mat interim = gblurr1dCh * colSlice.t();
            interim = interim.t();
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].col(offset + c));
        }
    }


    for(ch = 0 ; ch < 3 ; ch++)
    {
        procChannels[ch].copyTo(ogChannels[ch]);
    }


    for(ch = 0; ch < 3; ch++)
    {
        for(r = 0; r < (numRowsFrame - numColsFilter + 1); r++)
        {
            Mat rowSlice(ogChannels[ch].rowRange(r, r+numColsFilter)); 
            Mat interim(gblurr1dCh * rowSlice);
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].row(r));
        }
    }

    Mat outChannels[3];

    split(procFrame, outChannels);

    for(ch = 0 ; ch < 3 ; ch++)
    {
        
        Rect toroi(offset, offset, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        Rect fromroi(0, 0, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        procChannels[ch](fromroi).copyTo(outChannels[ch](toroi));
        outChannels[ch].convertTo(outChannels[ch], CV_8UC1);
    }

    merge(outChannels, 3, dst);

    return 1;
}



// This function is used to apply the Sobel X filter on the image/frame.
// The filter is first seperated into 2 filters, and then applied row-wise and column-wise respectively.
// The operations are computed channel by channel.
    
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
int sobelX3x3(Mat &src, Mat &dst )
{
    Mat procFrame = src.clone();

    float filterData1dRow[3] = {-1, 0, 1};
    float filterData1dCol[3] = {1, 2, 1};

    Mat filterRow = Mat(1, 3, CV_32F, filterData1dRow);

    Mat filterCol = Mat(3, 1, CV_32F, filterData1dCol);

    Size dimsFrame = procFrame.size();
    int numColsFrame = dimsFrame.width;
    int numRowsFrame = dimsFrame.height;

    Size dimsFilter = filterRow.size();
    int numColsFilter = dimsFilter.width;
    int numRowsFilter = dimsFilter.height;
    
    int r, c, ch; 

    Mat ogChannels[3], procChannels[3];

    procFrame.convertTo(procFrame, CV_32F);

    split(procFrame, procChannels);

    split(procFrame, ogChannels);
    
    int numiters = 0;

    int offset = numColsFilter / 2;

    int sumBlurr = 1;

    for(ch = 0; ch < 3; ch++)
    {
        for(c = 0; c < (numColsFrame - numColsFilter + 1); c++)
        {
            Mat colSlice = ogChannels[ch].colRange(c, c+numColsFilter); 
            Mat interim = filterRow * colSlice.t();
            interim = interim.t();
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].col(offset + c));
        }
    }

    for(ch = 0 ; ch < 3 ; ch++)
    {
        procChannels[ch].copyTo(ogChannels[ch]);
    }

    sumBlurr = 4;

    for(ch = 0; ch < 3; ch++)
    {
        for(r = 0; r < (numRowsFrame - numColsFilter + 1); r++)
        {
            Mat rowSlice(ogChannels[ch].rowRange(r, r+numColsFilter)); 
            Mat interim(filterCol.t() * rowSlice);
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].row(r));
        }
    }

    Mat outChannels[3];

    split(procFrame, outChannels);

    for(ch = 0 ; ch < 3 ; ch++)
    {
        Rect toroi(offset, offset, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        Rect fromroi(0, 0, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        procChannels[ch](fromroi).copyTo(outChannels[ch](toroi));
    }
    merge(outChannels, 3, dst); 
    dst.convertTo(dst, CV_16SC3);   
    return 1;
}

// This is a function used to Visualize the derivative of a frame/image. (More specifically, to visualize the output of the sobelX3x3, 
//     sobelY3x3 filters).

//    Arguments:
//     - *Mat src : A cv::Mat* (cv::Mat pointer) to the input frame.  
void vizderivative(Mat &src)
{
    
    src.convertTo(src, CV_32FC3);

    double minVal, maxVal;
    
    minMaxLoc(src, &minVal, &maxVal);
    src = (src - minVal) / (maxVal - minVal);
    
    src = src * 255;
    
    src.convertTo(src, CV_8UC3);
}


// This function is used to apply the Sobel Y filter on the image/frame.
// The filter is first seperated into 2 filters, and then applied row-wise and column-wise respectively.
// The operations are computed channel by channel.
    
//    Arguments:
//     - Mat &src : A cv::Mat reference to the input frame.
//     - Mat &dst : A cv::Mat reference to the destimation frame.
// 
//    returns:
//        - return an integer value of 1.
int sobelY3x3(Mat &src, Mat &dst )
{
    Mat procFrame = src.clone();

    // THE WHOLE FUNCTION IS THE SAME AS sobelX3x3, EXCEPT, THE filterData1dRow and filterData1dCol are interchanged/
    float filterData1dRow[3] = {1, 2, 1};
    float filterData1dCol[3] = {-1, 0, 1};

    Mat filterRow = Mat(1, 3, CV_32F, filterData1dRow);

    Mat filterCol = Mat(3, 1, CV_32F, filterData1dCol);

    Size dimsFrame = procFrame.size();
    int numColsFrame = dimsFrame.width;
    int numRowsFrame = dimsFrame.height;

    Size dimsFilter = filterRow.size();
    int numColsFilter = dimsFilter.width;
    int numRowsFilter = dimsFilter.height;
    
    
    int r, c, ch; 

    Mat ogChannels[3], procChannels[3];

    procFrame.convertTo(procFrame, CV_32F);

    split(procFrame, procChannels);

    split(procFrame, ogChannels);
    
    int numiters = 0;

    int offset = numColsFilter / 2;

    int sumBlurr = 4;

    for(ch = 0; ch < 3; ch++)
    {
        for(c = 0; c < (numColsFrame - numColsFilter + 1); c++)
        {
            Mat colSlice = ogChannels[ch].colRange(c, c+numColsFilter); 
            Mat interim = filterRow * colSlice.t();
            interim = interim.t();
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].col(offset + c));
        }
    }


    for(ch = 0 ; ch < 3 ; ch++)
    {
        procChannels[ch].copyTo(ogChannels[ch]);
    }

    sumBlurr = 1; 

    for(ch = 0; ch < 3; ch++)
    {
        for(r = 0; r < (numRowsFrame - numColsFilter + 1); r++)
        {
            Mat rowSlice(ogChannels[ch].rowRange(r, r+numColsFilter)); 
            Mat interim(filterCol.t() * rowSlice);
            interim = interim / sumBlurr;
            interim.copyTo(procChannels[ch].row(r));
        }
    }

    Mat outChannels[3];

    split(procFrame, outChannels);

    for(ch = 0 ; ch < 3 ; ch++)
    {
        Rect toroi(offset, offset, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        Rect fromroi(0, 0, numColsFrame - numColsFilter + 1, numRowsFrame - numColsFilter + 1);
        procChannels[ch](fromroi).copyTo(outChannels[ch](toroi));
    }

    merge(outChannels, 3, dst);
    dst.convertTo(dst, CV_16SC3);   
    return 1;
}


// This function is used to find the magnitude of the gradients of an image/video. It takes the outputs of the the sobelX3x3
// operation and the sobelY3x3 operation as inputs, and computes the L2 Norm between them.

    
//    Arguments:
//     - Mat &sx : A cv::Mat reference to the output of the sobelX3x3 operation.
//     - Mat &sy : A cv::Mat reference to the output of the sobelY3x3 operation.
//     - Mat &dst : A cv::Mat reference tp the output of the 

//    returns:
//        - return an integer value of 1.
int magnitude(Mat &sx, Mat &sy, Mat &dst)
{
    // dst.convertTo(dst, CV_16SC3);
    sx.convertTo(sx, CV_32FC3);
    sy.convertTo(sy, CV_32FC3);
    dst.convertTo(dst, CV_32FC3);

    Mat outChannels[3], xChannels[3], yChannels[3];

    split(dst, outChannels);
    split(sx, xChannels);
    split(sy, yChannels);

    int ch;
    for(ch = 0 ; ch < 3 ; ch++ )
    {
        outChannels[ch] = (yChannels[ch].mul(yChannels[ch])) + (xChannels[ch].mul(xChannels[ch])) ;
        sqrt(outChannels[ch], outChannels[ch]);
    }

    merge(outChannels, 3, dst);
    dst.convertTo(dst, CV_16SC3);

    return 1;
}


//
//The function blurs and quantizes an input image.
//
//  Arguements:
//      - frame:  The "frame" parameter is a reference to a Mat object, which represents an image frame.
//
//  returns:
//      - returns an integer value of 1.

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels )
{
    blur5x5_2(src, dst);
    int binSize = 255/levels;
    src.convertTo(dst, CV_16SC3);
    dst = dst / binSize;
    dst = dst * binSize;
    dst.convertTo(dst, CV_8UC3);
    return 1;
}

// This function detects faces in an input image and draws boxes around them. Detection smoothing is done by    
// averaging the last 2 detection.
//
// Arguments:
//   - frame: A reference to a Mat object representing an image frame.
//
// Returns:
//   - This function does not return a value.
void findFaces(Mat &frame)
{
    Mat grey;
    vector<cv::Rect> faces;
    Rect last(0, 0, 0, 0);

    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

    // detect faces
    detectFaces( grey, faces );

    // draw boxes around the faces
    drawBoxes( frame, faces );

    // add a little smoothing by averaging the last two detections
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
}

// This function converts an input image into a cartoon-style representation. The following operations are applied
//    to cartoonize the image: 
//    increase saturation by 90 -> apply bilateral filter -> apply canny filter -> find and draw contours.
//
// Arguments:
//   - frame: A reference to a Mat object representing the input image.
//
// Returns:
//   - This function modifies the 'frame' input parameter in-place and does not return a value.

void makecartoon(Mat &frame)
{   
    Mat original;
    frame.copyTo(original);
    cv::cvtColor( original, original, cv::COLOR_BGR2HSV, 0);
    Mat channels[3];
    split(original, channels);
    channels[1] = channels[1] + 90;
    merge(channels, 3, original);
    cv::cvtColor( original, original, cv::COLOR_HSV2BGR, 0);
    
    Mat dst;
    original.copyTo(dst);
    bilateralFilter(original, dst, 9, 75, 75, BORDER_DEFAULT);

    Mat src_gray;
    int thresh = 40;
    cvtColor( original, src_gray, COLOR_BGR2GRAY );
    blur( frame, src_gray, Size(3,3) );
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );

    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar(0, 0, 0);
        drawContours( dst, contours, (int)i, color, 1, LINE_8, hierarchy, 0 );
    }
    
    frame = dst;
    frame.convertTo(frame, CV_8UC3);
}

// This function applies a blur effect to the background in an input image but not to detected faces.
// It first creates a background by applying a Gaussian blur to the original frame. Then, it detects faces
// in the grayscale version of the input frame and overlays the original color information of the detected
// faces onto the blurred background. Finally, it draws boxes around the detected faces. Detection smoothing is done 
// by averaging the last 2 detection.
//
// Arguments:
//   - frame: A reference to a Mat object representing the input image.
//
// Returns:
//   - This function modifies the 'frame' input parameter in-place and does not return a value.
void envblurr(Mat &frame)
{
    Mat backGround = frame.clone();

    int ksize = 7;
    double sigma = 5;
    GaussianBlur(backGround, backGround, Size(ksize, ksize), sigma, sigma);

    // blur5x5_2(backGround, backGround);

    int ch;

    Mat grey;
    vector<cv::Rect> faces;
    Rect last(0, 0, 0, 0);

    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

    // detect faces
    detectFaces( grey, faces );

    Mat outChannels[3];
    Mat ogChannels[3];

    split(backGround, outChannels);
    split(frame, ogChannels);

    for(ch = 0; ch < 3; ch++)
    {
        int i;
        for(i = 0 ; i < faces.size() ; i++)
        {
            Rect face( faces[i] );
            ogChannels[ch](face).copyTo(outChannels[ch](face));
        }   
    }

    merge(outChannels, 3, frame);

    // draw boxes around the faces
    drawBoxes( frame, faces );

    // add a little smoothing by averaging the last two detections
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
}

// This function is used to obtain.
// It begins by creating a copy of the input frame to serve as the background, then applies a custom BW conversion 
// algorithm ('betterBGRtoBW') to the background image. Afterward, it detects faces in the grayscale version of the input 
// frame and overlays the original color information of the detected faces onto the BW background. Additionally, it enhances
// the contrast of the input frame using the 'ContrastSigmoid' function. Finally, the function draws boxes around the 
// detected faces.Detection smoothing is done by averaging the last 2 detection.
//
// Arguments:
//   - frame: A reference to a Mat object representing the input image.
//
// Returns:
//   - This function modifies the 'frame' input parameter in-place and does not return a value.

void bwenv(Mat &frame)
{
    Mat backGround = frame.clone();
    
    // int ksize = 7;
    // double sigma = 5;
    // GaussianBlur(backGround, backGround, Size(ksize, ksize), sigma, sigma);

    betterBGRtoBW(backGround, backGround);

    int ch;

    Mat grey;
    vector<cv::Rect> faces;
    Rect last(0, 0, 0, 0);

    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

    // detect faces
    detectFaces( grey, faces );

    Mat outChannels[3];
    Mat ogChannels[3];

    split(backGround, outChannels);

    ContrastSigmoid(frame, frame);
    split(frame, ogChannels);

    for(ch = 0; ch < 3; ch++)
    {
        int i;
        for(i = 0 ; i < faces.size() ; i++)
        {
            Rect face( faces[i] );
            ogChannels[ch](face).copyTo(outChannels[ch](face));
        }   
    }

    merge(outChannels, 3, frame);

    // draw boxes around the faces
    drawBoxes( frame, faces );

    // add a little smoothing by averaging the last two detections
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
}

// This function creates a negative image effect from the input image by inverting the colors of each channel.
// It converts the input frame to a floating-point representation, negates each color channel, and adds 255 to each channel's values
// to create the negative effect. Then, it converts the result back to 8-bit unsigned chars. 
//
// Arguments:
//   - frame: A reference to a Mat object representing the input image.
//
// Returns:
//   - This function modifies the 'frame' input parameter in-place and does not return a value.
void makeNegative(Mat &frame)
{
    frame.convertTo(frame, CV_32FC3);

    Mat outChannels[3];
    
    split(frame, outChannels);

    int i = 0;
    for(i = 0 ; i < 3 ; i++)
    {
        outChannels[i] = (- outChannels[i]) + 255;
        outChannels[i].convertTo(outChannels[i], CV_8U);
    }
    
    merge(outChannels, 3, frame);   

    frame.convertTo(frame, CV_8UC3);
}