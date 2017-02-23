#ifndef DetectorsValidationNightFox
#define DetectorsValidationNightFox

#ifndef MacroNightFox
#include "fcns.cpp"
#endif

#ifndef ValidationFcnsNightFox
#include "validating_fcns.cpp"
#endif

#define CONFIDENCE_LBPH_TH 85.0
#define CONFIDENCE_FISHER_TH 3000.0
#define CONFIDENCE_EIGEN_TH 500
#define CONFIDENCE_LBPH_UP_TH 115.0

#define STATE_LBPH_CONF 1
#define STATE_FISHER_CONF 2
#define STATE_EIGEN_CONF 3
#define STATE_MAYBE 4
#define STATE_UNKNOWN 5

class valid_detectors {
	public:
        valid_detectors(int stable_out, int buffer_size);
        bool fill_buffer(int out_lbph, double lbph_confidence, int out_fisher, double fisher_confidence, int out_eigen, double eigen_confidence);
        void validate(int (*validateFcn)(std::vector<int>));
        bool debounce();
        int get_sure_out();
        void clear();   // Argument to be defined
        int detectors;
        int max_buffer_size;
        // Stable / Debounce out
        int stable_cnt;
        int stable_out;
        int last_out;   // Out of i-1
        // Sure output
        int last_sure_out;
		//Dict y_map;
    private:
        std::vector<int> guesses_lbph;
        std::vector<double> guesses_lbph_confidence;
        std::vector<int> guesses_fisher;
        std::vector<double> guesses_fisher_confidence;
        std::vector<int> guesses_eigen;
        std::vector<double> guesses_eigen_confidence;
        // Validated output. To be checked if stable
        int out_LBPH;
        double avg_lbph_conf;
        int out_EIGEN;
        double avg_eigen_conf;
        int out_FISHER;
        double avg_fisher_conf;
        // State Debug
        int state;
};

valid_detectors::valid_detectors(int stable_out=1, int buffer_size=5)
{
    this->stable_out = stable_out; 
    this->max_buffer_size = buffer_size;
    this->stable_cnt = 0;
    this->last_sure_out = UNKNOWN_OUT;
}

bool valid_detectors::fill_buffer(int out_lbph, double lbph_confidence, int out_fisher, double fisher_confidence, int out_eigen, double eigen_confidence )
{
    guesses_lbph.push_back(out_lbph);
    guesses_lbph_confidence.push_back(lbph_confidence);
    guesses_fisher.push_back(out_fisher);
    guesses_fisher_confidence.push_back(fisher_confidence);
    guesses_eigen.push_back(out_eigen); // Checking if EIGEN is causing issues
    guesses_eigen_confidence.push_back(eigen_confidence);
    // Check if buffers are full
    if(this->guesses_lbph.size() == this->max_buffer_size)
    {
        return true;
    }
    return false;
}

void valid_detectors::clear()
{
    this->guesses_lbph.clear();
    this->guesses_lbph_confidence.clear();
    this->guesses_eigen.clear();
    this->guesses_fisher.clear();
    this->guesses_fisher_confidence.clear();
    this->stable_cnt = 0;
}

void valid_detectors::validate(int (*validateFcn)(std::vector<int>))
{
    // Checking LBPH guesses
    this->out_LBPH = validateFcn(this->guesses_lbph);
    this->avg_lbph_conf = avg_value(this->guesses_lbph_confidence);
    // Checking EIGEN guesses
    this->out_EIGEN = validateFcn(this->guesses_eigen);
    this->avg_eigen_conf = avg_value(this->guesses_eigen_confidence);
    // Check FISHER guesses
    this->out_FISHER = validateFcn(this->guesses_fisher);
    this->avg_fisher_conf = avg_value(this->guesses_fisher_confidence);
}

bool valid_detectors::debounce()
{
    int c_out = this->get_sure_out();
    if(this->last_out == c_out)
    {
        this->stable_cnt++;
    }else{
        this->stable_cnt = 0;
    }
    std::cout << this->stable_cnt << " stable cnt" << std::endl;
    // Update last out
    this->last_out = c_out;
    // Made it so far. Everything is fine and beautiful. Check if it is stable
    if(this->stable_cnt >= this->stable_out)
    {
        std::cout << "STATE -> " << this->state << std::endl;
        this->last_sure_out = c_out;
        return true;
    }
    return false;
}

int valid_detectors::get_sure_out()
{
    //Check if all 3 detectors have same output
    int output;
    /*
    std::cout << "LBPH " << this->out_LBPH << " - " << this->avg_lbph_conf << std::endl;
    std::cout << "FISHER " << this->out_FISHER << " - " << this->avg_fisher_conf << std::endl;
    std::cout << "EIGEN " << this->out_EIGEN << " - " << this->avg_eigen_conf << std::endl;
    */
    if(this->avg_lbph_conf < CONFIDENCE_LBPH_TH)
    {
        this->state = STATE_LBPH_CONF;
        output = this->out_LBPH;
    }else if(this->avg_fisher_conf < CONFIDENCE_FISHER_TH)
    {
        this->state = STATE_FISHER_CONF;
        output = this->out_FISHER;
    }else if(this->avg_eigen_conf < CONFIDENCE_EIGEN_TH)
    {
        this->state = STATE_EIGEN_CONF;
        output = this->out_EIGEN;
    }else if(this->out_LBPH == this->out_FISHER && this->out_LBPH == this->out_EIGEN)
    {
        this->state = STATE_MAYBE;
        output = out_LBPH;
    }else{
        this->state = STATE_UNKNOWN;
        output = UNKNOWN_OUT;
    }
    return output;
    
    
    /*if(this->out_LBPH == this->out_FISHER && this->out_LBPH == this->out_EIGEN)
    {
        output = this->out_LBPH;
        this->state = STATE_ALL_AGREE;
    }else if(this->avg_lbph_conf < CONFIDENCE_TH){
        // Check if LBPH is below TH
        output = this->out_LBPH;
        this->state = STATE_LBPH_CONF;
    }else if(this->out_LBPH == this->out_FISHER || this->out_LBPH == this->out_EIGEN){
        // 2 out of 3 agree
        output = this->out_LBPH; //this->out_EIGEN;
        this->state = STATE_LOW_CONF;
    }else if(this->out_FISHER == this->out_EIGEN){
        output = this->out_FISHER;
        this->state = STATE_LOW_CONF;
    }else{
        output = UNKNOWN_OUT;
        this->state = STATE_UNKNOWN;
    } */
}


#endif
