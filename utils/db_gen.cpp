#ifndef DatabaseNightFox
#define DatabaseNightFox

#ifndef MacroNightFox
#include "fcns.cpp"
#endif

#include <boost/filesystem.hpp>

// Slash delimiter that is OS dependant. For some reason , using boost filesystem return \ if WIN32, / if Linux
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static const char slash='\\';
#else
static const char slash='/';
#endif


class database_gen {
	public:
        database_gen(std::string faces_dp);
        void print_db();
        int getClassID(const std::string& name);
		std::vector< cv::Mat > X_LBPH;
		cv::Mat y;
		int index;	// Keeps track on how many labels are inside Dict var
		std::set<std::string> y_map;
};


database_gen::database_gen(std::string faces_dp)
{
    boost::filesystem::path c_dir(faces_dp);
	for (boost::filesystem::recursive_directory_iterator iter(c_dir), end; iter != end; ++iter)
	{
		std::string full_path = iter->path().string();
        std::cout << "File found -> " << full_path << std::endl;
		// Check if current file is a jpg
		std::string::size_type pos_end;
		pos_end = full_path.find("jpg", 0);
		if ( pos_end != std::string::npos) {	// JPG File found!
            //std::cout << "JPG Found!" << std::endl;
			// Split full_path
			std::vector < std::string > dirs = split(full_path, slash);	// Char '', string ""
			std::string name = dirs[dirs.size() - 2];   // One before last value, being the name of the dir
			this->y_map.insert(name);
			// Processing image and pass it to X and y
			cv::Mat img = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
			//cv::equalizeHist(img, img);
			cv::GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);
			cv::Mat img_res;
			cv::resize(img, img_res, DIM_REF, cv::INTER_CUBIC);
			this->X_LBPH.push_back(img_res);
            //this->X_HOG.push_back(HOG_COMPUTE(img_res, CV_64F, 16, 2));	// CV_32FC1
			this->y.push_back(this->getClassID(name));			  
		}
	}
    // Print out created DB
    this->print_db();
}

int database_gen::getClassID(const std::string& name)
{
    int index = 0;
    for(auto it = this->y_map.begin(); it != this->y_map.end(); ++it)
    {
        if(*it == name) break;
        ++index;
    }
    return index;
}


void database_gen::print_db()
{
    int idx = 0;
    for(std::set<std::string>::iterator iter = this->y_map.begin(); iter != this->y_map.end(); iter++)
    {
        std::cout << *iter << " -> " << idx++ << std::endl;
    }
}

#endif