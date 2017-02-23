#ifndef MacroNightFox
#define MacroNightFox

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;

#define DIM_REF Size(200, 200)


std::vector < std::string > split(std::string str, char delim)
{
	std::vector< std::string > elems;
	std::stringstream ss(str);
	std::string elem;
	while(std::getline(ss, elem, delim))
	{
		elems.push_back(elem);
	}
	return elems;
}

// OPENCV RELATED

template <typename Tp>
cv::Mat vector_to_mat(std::vector< std::vector<Tp> > &in_vector)
{
	// Create a Mat without rows. Initialize only columns number
	cv::Mat out(0, in_vector[0].size(), cv::DataType<Tp>::type);
	// Populate one row at the time
	for (int i=0; i< in_vector.size(); i++)
	{
		cv::Mat temp(1, in_vector[0].size(), cv::DataType<Tp>::type, in_vector[i].data());
		out.push_back(temp);
	}
	return out;
}

template <typename Tp>
cv::Mat vector_to_mat(std::vector<Tp> &in_vector)
{
    cv::Mat out(1, in_vector.size(), cv::DataType<Tp>::type, in_vector.data());
    return out;
}


cv::Mat HOG_COMPUTE(cv::Mat img, int ddepth=CV_8UC1, int bin_n = 16, int div_n = 2)
{
    std::vector<float> hist;
    //std::cout << "Width: " << img.cols << "\nHeight: " << img.rows << std::endl;
    
    /*
    cv::Size win_size = DIM_REF;
    cv::Size block_size = cv::Size(DIM_REF.width / div_n, DIM_REF.height / div_n);
    cv::Size block_stride = cv::Size(block_size.width / div_n, block_size.height / div_n);
    cv::Size cell_size = block_stride;
    HOGDescriptor hog(win_size, block_size, block_stride, cell_size, bin_n);
    hog.compute(img, hist);
    */
    
    cv::Size win_size = cv::Size(64, 64);
    cv::Size block_size = cv::Size(16, 16);
    cv::Size block_stride = cv::Size(8, 8);
    cv::Size cell_size = block_stride;
    HOGDescriptor hog(win_size, block_size, block_stride, cell_size, 9, 1, 4.0, HOGDescriptor::L2Hys, 0.2, false, 64);
    std::vector<cv::Point> locations;
    locations.push_back(cv::Point(100,100));
    hog.compute(img, hist, win_size, block_stride, locations);
    
    std::cout << "Hog Features Size:" << hist.size() << std::endl;    // 576
    cv::Mat out(1, hist.size(), cv::DataType<float>::type, *hist.data());
    //cv::normalize(out, out, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return out;
}

#endif