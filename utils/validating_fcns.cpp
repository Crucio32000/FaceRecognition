#ifndef ValidationFcnsNightFox
#define ValidationFcnsNightFox

bool validate_with_conf(std::vector<int> guesses, std::vector<double> confs)
{
    return true;
}

int validate_and(std::vector<int> guesses)
{
    for(int i=0; i<guesses.size(); i++)
    {
        if(guesses[0] == guesses[i])
        {
            continue;
        }else{
            return UNKNOWN_OUT;
        }
    }
    return guesses[0];
}


int validate_or(std::vector<int> guesses)
{
    for(int i=0; i<guesses.size(); i++)
    {
        for(int j=0; j<guesses.size(); j++)
        {
            if(guesses[i] == guesses[j])
            {
                return guesses[i];
            }
        }
    }
    return UNKNOWN_OUT;
}

template <typename T>
T avg_value(const std::vector<T> &in_vec)
{
    T avg = 0;
    for (int x : in_vec) avg += x;
    
    return in_vec.empty() ? 0 : avg/in_vec.size();
}
#endif
