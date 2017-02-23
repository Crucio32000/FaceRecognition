#ifndef FaceDetectorEIGENNightFox
#define FaceDetectorEIGENNightFox

#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp> 

typedef std::map< std::string, int > Dict;


class FaceDetectorEIGEN {
	public:
		FaceDetectorEIGEN(std::string model_fp);
		FaceDetectorEIGEN(const std::vector< cv::Mat >& input_dataset, const cv::Mat &out_dataset, const Dict &out_labels);
		void common_init();
        void init_labels(std::string label_fp);
		int guess(cv::Mat face);
		std::string get_label(int guess);
		void save_model();
        double confidence;
	private:
		cv::Ptr<cv::face::FaceRecognizer> model;
		Dict vocabulary;
};

FaceDetectorEIGEN::FaceDetectorEIGEN(const std::vector< cv::Mat >& input_dataset, const cv::Mat &out_dataset, const Dict &out_labels)
{
    // Initiating Model
    this->model = cv::face::createEigenFaceRecognizer(100);
	//this->model = cv::face::createEigenFaceRecognizer((int)out_labels.size()*10); // NumOfComponents. 80 was suggested
    // Training SVM
	this->model->train(input_dataset, out_dataset);
	this->vocabulary = out_labels;   
}


FaceDetectorEIGEN::FaceDetectorEIGEN(std::string model_fp)
{
    // Loading Model
    this->model = cv::face::createEigenFaceRecognizer();
	this->model->load(model_fp.c_str()); 
    // Load Labels
    this->init_labels("y_maps_FISHER.txt");
}


std::string FaceDetectorEIGEN::get_label(int guess)
{
    Dict::iterator it = this->vocabulary.begin();
    for(Dict::iterator it = this->vocabulary.begin(); it != this->vocabulary.end(); it++)
    {
        if(it->second == guess)
        {
            return it->first;
        }
    }
    return UNKNOWN_LABEL;
}



int FaceDetectorEIGEN::guess(cv::Mat face)
{
    // Resize to reduce computation time / Adapt to HOG
    //resize(face, face_res, DIM_REF, INTER_LINEAR);
    // Getting output
	int out;
    this->model->predict(face, out, this->confidence);
	return out;   
}


void FaceDetectorEIGEN::save_model()
{
    this->model->save("BEST_EIGEN.XML");
	// Saving mappings
	std::ofstream outf("y_maps_EIGEN.txt");
	if (outf)
	{
        int idx = 0;
		for(Dict::iterator iter = this->vocabulary.begin(); iter!= this->vocabulary.end(); iter++)
		{
			outf << iter->first << ":" << iter->second << std::endl;
		}
	}
	std::cout << "Model saved successfully!" << std::endl;
}


void FaceDetectorEIGEN::init_labels(std::string label_fp)
{
    // Something that loads vocabulary (name:index)
    std::string bstring;
    std::ifstream buff;
    buff.open(label_fp.c_str());
    while(!buff.eof())
    {	// Name:Index
        std::getline(buff, bstring);
        std::vector< std::string > temp = split(bstring, ':');
        if (temp.size()  == 2)
        {
            int index=0;
            sscanf(temp[1].c_str(), "%d", &index);
            this->vocabulary[temp[0]] = index;
        }
    }
}








#endif