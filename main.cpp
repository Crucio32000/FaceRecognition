#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <math.h>

#include <boost/filesystem.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include <opencv2/opencv.hpp> 
#include <opencv2/face.hpp>



#define PI_2 M_PI*2

#define FONT    FONT_HERSHEY_PLAIN
#define FONT_COLOR  cv::Scalar(255, 255, 255)
#define SURE_COLOR cv::Scalar(255, 255, 0)
#define LINE_TYPE   CV_AA
#define LINE_SPACING 20

#define UNKNOWN_INT 300
#define UNKNOWN_LABEL   std::string("Unknown")

#include "utils/fcns.cpp"
#include "utils/db_gen.cpp"
#include "utils/validator.cpp"
#include "utils/validating_fcns.cpp"

#include "lbph/lbph.cpp"
//#include "eigen/eigen.cpp"
#include "fisher/fisher.cpp"


cv::Mat equalizeIntensity_1(const cv::Mat& c_frame)
{
    if(c_frame.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(c_frame,ycrcb,CV_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,result, CV_YCrCb2BGR);
        return result;
    }else{
        cv::Mat result;
        cv::equalizeHist(c_frame, result);
        return result;
    }   
}

cv::Mat equalizeIntensity_2(const cv::Mat& c_frame)
{
    cv::Mat result;
    if(c_frame.channels() >= 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(c_frame, channels);
        for(int i=0; i< channels.size(); i++)
        {
            cv::equalizeHist(channels[i], channels[i]);
            
        }
        cv::merge(channels, result);
    }else{
        cv::equalizeHist(c_frame, result);
    }
    
    return result;
}


void get_faces(std::vector<cv::Rect_<int>> &faces, cv::Mat &frame)
{
    // Init detector and convert mat to dlib format
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::array2d<dlib::bgr_pixel> cimg;
    dlib::assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(frame));
    // Initialize Pose detector
    //dlib::shape_predictor sp;
    //dlib::deserialize("./utils/shape_predictor_68_face_landmarks.dat") >> sp;
    // Detect faces
    std::vector<dlib::rectangle> det_faces = detector(cimg);
    // Convert rectangle to cv::rec
    for(int i=0; i<det_faces.size(); i++)
    {
        // -- Detect landmarks for pose extimation
        dlib::rectangle c_face_dlib = det_faces[i];
        /*dlib::full_object_detection c_face_lm = sp(cimg, c_face_dlib);
        // -- Detect Pose
        dlib::point left_eye = c_face_lm.part(39);
        dlib::point right_eye = c_face_lm.part(43);
        dlib::point nose = c_face_lm.part(27);
        double left_dist = sqrt( pow((left_eye.x() - nose.x()), 2) + pow((left_eye.y() - nose.y()), 2) );
        double right_dist = sqrt( pow((right_eye.x() - nose.x()), 2) + pow((right_eye.y() - nose.y()), 2) );
        if(left_dist > 1.1*right_dist || left_dist < 0.9*right_dist)
        {
            continue;
        }*/
        
        // -- Convert dlib::rectangle to cv::Rect_ 
        cv::Rect_<int> roi(c_face_dlib.left(), c_face_dlib.top(), c_face_dlib.width(), c_face_dlib.height());
        // If roi goes out of scope(the main matrix, then it gives back error because of the condition below)
        if (roi.x + roi.width > frame.cols || roi.y + roi.height > frame.rows || roi.x < 0 || roi.y < 0)
        {
            continue;
        }
        faces.push_back(roi);
    }
}

bool pose_est(dlib::array2d<dlib::bgr_pixel> cimg, dlib::rectangle)
{
    return true;
}


int stored_frames = 0;
void store_face(cv::Mat face)
{
    std::cout << "Storing face @";
    std::string filename;
    std::stringstream ss;
    ss << stored_frames++;
    filename = "./Unknown/" +  ss.str() + ".jpg";
    std::cout << filename <<std::endl;
    cv::imwrite(filename, face);
}

void store_update_face(cv::Mat face, std::string name)
{
    std::cout << "Storing face @";
    std::string filename;
    std::stringstream ss;
    ss << stored_frames++;
    filename = "./Training/" + name + "_" +  ss.str() + ".jpg";
    std::cout << filename <<std::endl;
    cv::imwrite(filename, face);
}


bool train_model = false;

int main(int argc, char ** argv)
{
    // Allocating memory for Facedetectors
    //FaceDetectorEIGEN * detector_eigen;
    FaceDetectorLBPH * detector_lbph;
    FaceDetectorFISHER * detector_fisher;
    if (argc >= 4)
    {
        std::cout << "Initializing FaceDetectors" << std::endl;
        // Init HOG
        //std::cout << "HOG Detector Init..." << argv[1] << std::endl;
        //detector_eigen = new FaceDetectorEIGEN((std::string) argv[1]);
        // Init LBPH
        std::cout << "LBPH Detector Init..." << argv[2] << std::endl;
        detector_lbph = new FaceDetectorLBPH((std::string) argv[2]);
        // Init FISHER
        std::cout << "FISHER Detector Init..." << argv[3] << std::endl;
        detector_fisher = new FaceDetectorFISHER((std::string) argv[3]);
    } else{
        // Init db_generator
        database_gen * db_gen = new database_gen("faces_noise");
        std::cout << "Training Face Detectors!" << std::endl;
        // Init FaceDetectors
        //std::cout << "HOG Detector Train..." << std::endl;
        //detector_eigen = new FaceDetectorEIGEN(db_gen->X_LBPH, db_gen->y, db_gen->y_map);
        std::cout << "LBPH Detector Train..." << std::endl;
        detector_lbph = new FaceDetectorLBPH(db_gen->X_LBPH, db_gen->y, db_gen->y_map);
        std::cout << "FISHER Detector Train..." << std::endl;
        detector_fisher = new FaceDetectorFISHER(db_gen->X_LBPH, db_gen->y, db_gen->y_map);
        delete db_gen;
        train_model = true;
    }
    // Initializing Validator (stable CNT and buffer size)
    valid_detectors * validator;
    validator = new valid_detectors(5);
    // Initializing common variables
    //VideoCapture cap("http://192.168.0.250:2020/stream?topic=/stereo/image_raw?quality=100");
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_BUFFERSIZE,3);
    if (!cap.isOpened()) {
        std::cout << "Could not open camera!" << std::endl;
    }
    // Init vectors
    std::vector< cv::Rect_<int> > faces;
    cv::Mat frame;
    cv::Mat color_frame;
    // Related to user-triggered update
    int output_update = 0;
    cv::Mat face_res;
    
    // Training or Testing
    std::cout << "Testing FaceDetector. Stand Still in front of the camera!" << std::endl;
    while(true)
    {
        cap >> frame;   // Equivalent to cap.read() in Python
        //frame = equalizeIntensity_2(frame);
        color_frame = frame;
        get_faces(faces, frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        process_frame(frame);
        if(faces.size() > 0)
        {
            for(int i=0; i<faces.size(); i++)
            {
                // If eyes are not detected inside face, then iterate through next 
                cv::Mat face = frame(faces[i]);
                //Get guesses from Detectors
                // First resize the image
                cv::resize(face , face_res, DIM_REF, cv::INTER_CUBIC);
                //cv::normalize(face_res, face_res, 0, 255, cv::NORM_MINMAX);
                int output_LBPH = detector_lbph->guess(face_res);
                //int output_EIGEN = detector_eigen->guess(face_res);
                int output_FISHER = detector_fisher->guess(face_res);
                //std::string label_EIGEN_str = detector_eigen->get_label(output_EIGEN);
                std::string label_LBPH_str = detector_lbph->get_label(output_LBPH);
                std::string label_FISHER_str = detector_fisher->get_label(output_FISHER);
                char label_LBPH[30];
                sprintf(label_LBPH, "%s - %.2g", label_LBPH_str.c_str(), detector_lbph->confidence);
                char label_FISHER[30];
                sprintf(label_FISHER, "%s - %.2g", label_FISHER_str.c_str(), detector_fisher->confidence);
                //char label_EIGEN[30];
                //sprintf(label_EIGEN, "%s - %.2g", label_EIGEN_str.c_str(), detector_eigen->confidence);
                // Wait for validator to fill its buffer and return a sure result
                // Get sure answer or unknown if every check fails
                int output_sure;
                std::string label_sure;
                if(validator->fill_buffer(output_LBPH, detector_lbph->confidence, output_FISHER, detector_fisher->confidence))
                {
                    validator->validate(validate_and);
                    output_sure = validator->get_sure_out();
                    // -- Set label according to output of validator
                    if(output_sure == UNKNOWN_INT)
                    {
                        //Store actual face
                        store_face(color_frame(faces[i]));
                        label_sure = UNKNOWN_LABEL;
                    }else{
                        label_sure = detector_lbph->get_label(output_sure);
                    }
                    std::cout << label_sure << std::endl;
                    // -- Update LBPH if FISHER returns something below threshold and their output matches
                    if(validator->can_update())
                    {
                        detector_lbph->update(face_res, output_FISHER);
                    }
                    //Clear buffer
                    validator->clear();
                } 
                //Write guesses on current frame
                cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
                cv::Point label_lbph_p( faces[i].x , faces[i].y + faces[i].height );
                //cv::Point label_hog_p( faces[i].x , faces[i].y + faces[i].height+ LINE_SPACING );
                cv::Point label_fisher_p( faces[i].x , faces[i].y + faces[i].height+ LINE_SPACING*2 );
                cv::Point label_sure_p( faces[i].x , faces[i].y + faces[i].height+ LINE_SPACING );
                cv::ellipse(color_frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 10*i, 255-10*i ), 4, 8, 0 );
                cv::putText(color_frame, label_LBPH, label_lbph_p, FONT, 2, FONT_COLOR, 1, LINE_TYPE);
                //cv::putText(color_frame, label_EIGEN, label_hog_p, FONT, 2, FONT_COLOR, 1, LINE_TYPE);
                cv::putText(color_frame, label_FISHER, label_fisher_p, FONT, 2, FONT_COLOR, 1, LINE_TYPE);
                cv::putText(color_frame, label_sure.c_str(), label_sure_p, FONT, 2, SURE_COLOR, 1, LINE_TYPE);
            }
            // Clear vector
            faces.clear();
        }else{
            //continue;
        }
        // Show current frame
        cv::imshow("Cam", color_frame);
        char key_p = cv::waitKey(1);
        if(key_p == 'q')
        {
            cv::destroyWindow("Cam");
            break;
        }else if(key_p == 's')
        {
            store_face(face_res);
        }else{
            output_update = key_p - '0';    // or - 48 since ASCII starts from this value
            //std::cout << "Pressed -> " << output_update << std::endl;
            if(output_update >= 0)
            {
                detector_lbph->update(face_res, output_update);
                store_update_face(face_res, detector_lbph->get_label(output_update));
            }
        }
    }
    
    // Save model if it has been trained. Otherwise, just continue executing
    if(train_model)
    {
        std::cout << "Are current detectors good enough ?" << std::endl;
        int choice;
        std::cin >> choice;
        if(choice>0)
        {
            //detector_eigen->save_model();
            detector_lbph->save_model();
            detector_fisher->save_model();
        }
    }
    //delete detector_eigen;
    delete detector_lbph;
    delete detector_fisher;
    delete validator;
    return true;
}



