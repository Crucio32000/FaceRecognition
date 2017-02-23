#ifndef FaceDetectorLBPHNightFox
#define FaceDetectorLBPHNightFox

#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp> 

#define LBPH_RADIUS 1   //3
#define LBPH_NEIGHBORS 8
#define LBPH_GRID_X 8
#define LBPH_GRID_Y 8
#define LBPH_TH 100.0


class FaceDetectorLBPH {
	public:
		FaceDetectorLBPH(std::string model_fp);
		FaceDetectorLBPH(const std::vector< cv::Mat > &input_dataset, const cv::Mat &out_dataset, const Dict &out_labels);
		void common_init();
        void init_labels(std::string label_fp);
		int guess(cv::Mat face);
		std::string get_label(int guess);
        void update(const cv::Mat &face, const int &guess);
		void save_model();
        double confidence;
	private:
		cv::Ptr<cv::face::FaceRecognizer> model;
		Dict vocabulary;
};

FaceDetectorLBPH::FaceDetectorLBPH(const std::vector< cv::Mat > &input_dataset, const cv::Mat &out_dataset, const Dict &out_labels)
{
    // Initiating Model
    this->model = cv::face::createLBPHFaceRecognizer(LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y, LBPH_TH);
    // Training SVM
	this->model->train(input_dataset, out_dataset);
	this->vocabulary = out_labels;   
}


FaceDetectorLBPH::FaceDetectorLBPH(std::string model_fp)
{
    // Loading Model
    this->model = cv::face::createLBPHFaceRecognizer(LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y, LBPH_TH);
	this->model->load(model_fp.c_str()); 
    // Load Labels
    this->init_labels("y_maps_LBPH.txt");
}

void FaceDetectorLBPH::update(const cv::Mat &face, const int &guess)
{
    std::cout << " UPDATING LBPH " << std::endl;
    std::vector<cv::Mat> input;
    input.push_back(face);
    cv::Mat output;
    output.push_back(guess);
    this->model->update(input,output);
}


std::string FaceDetectorLBPH::get_label(int guess)
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



int FaceDetectorLBPH::guess(cv::Mat face)
{
    // Getting output
	int out;
    this->model->predict(face, out, this->confidence);
	return out;   
}


void FaceDetectorLBPH::save_model()
{
    this->model->save("BEST_LBPH.XML");
	// Saving mappings
	std::ofstream outf("y_maps_LBPH.txt");
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


void FaceDetectorLBPH::init_labels(std::string label_fp)
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