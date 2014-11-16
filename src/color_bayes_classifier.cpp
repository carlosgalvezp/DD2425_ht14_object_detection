#include <object_detection/color_bayes_classifier.h>

Color_Bayes_Classifier::Color_Bayes_Classifier()
{
    read_models();
}

void Color_Bayes_Classifier::read_models()
{
    std::ifstream file;
    std::string models_path = std::string(getenv("HOME")) + models_rel_path +"models.txt";
    file.open(models_path.c_str());

    int n_colors;
    file >> n_colors;
    for(unsigned int i = 0; i < n_colors; ++i)
    {
        Color_Model model;
        file >> model.name >> model.h_mu >> model.h_sigma >> model.s_mu >> model.s_sigma;

        Model_Constants m_constants;
        m_constants.log_alpha = log(1.0/7.0); //Uniform distribution, change this!!
        m_constants.log_sigma_h = log(model.h_sigma);
        m_constants.log_sigma_s = log(model.s_sigma);
        m_constants.inv_2_sigma_h_2 =  1.0/(2* model.h_sigma * model.h_sigma);
        m_constants.inv_2_sigma_s_2 =  1.0/(2* model.s_sigma * model.s_sigma);
        m_constants.th_h_mahalanobis = TH_MAHALANOBIS * model.h_sigma;
        m_constants.th_s_mahalanobis = TH_MAHALANOBIS * model.s_sigma;

        std::cout << "Reading model "<<model.name<<std::endl;
        models_.push_back(model);
        models_constants_.push_back(m_constants);
    }
}

int Color_Bayes_Classifier::classify(const int &h, const int &s)
{
    // ** Compute MAP estimate given that the point is close enough to the corresponding distribution
    int max_class = -1;
    double max_log_likelihood = -std::numeric_limits<double>::infinity();

    for(std::size_t i = 0; i <  models_.size(); ++i)
    {
        const Color_Model &cm = models_[i];
        const Model_Constants &constants = models_constants_[i];

        // ** Check Mahalanobis distance
        double d_h = RAS_Utils::mahalanobis_distance(h, cm.h_mu, cm.h_sigma);
        double d_s = RAS_Utils::mahalanobis_distance(s, cm.s_mu, cm.s_sigma);

        if(d_h > constants.th_h_mahalanobis || d_s > constants.th_s_mahalanobis)
            continue;

        // ** Compute log-likelihood
        double g = discriminant(h, s, i);

        // ** Check if its maximum
        if( g > max_log_likelihood)
        {
            max_log_likelihood = g;
            max_class = i;
        }
    }
    return max_class;
}

double Color_Bayes_Classifier::discriminant(const double &h, const double &s, const int &idx)
{
    const Color_Model &cm = models_[idx];
    const Model_Constants & constants = models_constants_[idx];

    return    constants.log_alpha - constants.log_sigma_h - (h - cm.h_mu)*(h - cm.h_mu)*(constants.inv_2_sigma_h_2)
                                  - constants.log_sigma_s - (s - cm.s_mu)*(s - cm.s_mu)*(constants.inv_2_sigma_s_2);
}
