#ifndef NORMAL_LIKELIHOOD_H
#define NORMAL_LIKELIHOOD_H

#include <cmath>
#include <Eigen/Core>
#include <Eigen/LU>

class NormalLikelihood
{

public:
    explicit NormalLikelihood(const float sigma)
    {
        normalize_= 1.0 / std::sqrt(2.0 * M_PI * sigma * sigma);
        sigma_sq_ = sigma * sigma * 2.0;
    }
    float operator()(const float x) const
    {
        return normalize_* expf(-x* x/ sigma_sq_);
    }

protected:
    float normalize_;
    float sigma_sq_;
};

#endif
