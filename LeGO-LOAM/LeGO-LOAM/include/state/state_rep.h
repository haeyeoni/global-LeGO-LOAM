#ifndef STATE_REP_H
#define STATE_REP_H

#include <state/quat.h>
#include <state/vec3.h>

class PoseState 
{
public:

    PoseState()
    {
        diff_ = false;
    }
    PoseState(const Vec3 pose, const Quat rot)
    {
        pose_ = pose;
        rot_ = rot;
        diff_ = false;
    }

    PoseState(const Vec3 pos, const Vec3 rpy)
    {
        pose_ = pos;
        // rpy_ = RPYVec(rpy);
        rot_ = Quat(rpy);
        diff_ = true;
    }

    int size() const
    {
        return 7;
    }

    template <typename PointType> 
    void transform(pcl::PointCloud<PointType>& pc) const
    {
        const auto r = rot_.normalized();
        for (auto& p : pc.points)
        {
            const Vec3 t = r * Vec3(p.x, p.y, p.z) + pose_; // transformed
            p.x = t.x_;
            p.y = t.y_;
            p.z = t.z_;
        }
    }

    void normalize()
    {
        rot_.normalize();
    }

    float operator[](const size_t i) const
    {
        switch (i)
        {
            case 0: return pose_.x_;
            case 1: return pose_.y_;
            case 2: return pose_.z_;
            case 3: return rot_.x_;
            case 4: return rot_.y_;
            case 5: return rot_.z_;
            case 6: return rot_.w_;
            default:
                assert(false);
        }
        return 0;
    }

    float& operator[](const size_t i) 
    {
        switch (i)
        {
            case 0: return pose_.x_;
            case 1: return pose_.y_;
            case 2: return pose_.z_;
            case 3: return rot_.x_;
            case 4: return rot_.y_;
            case 5: return rot_.z_;
            case 6: return rot_.w_;
            default:
                assert(false);
        }
        return pose_.x_;
    }


    PoseState operator+(const PoseState& a) const
    {
        PoseState in = a;
        PoseState ret;
        for (size_t i = 0; i < 13; i++)
        {
        if (3 <= i && i <= 6)
            continue;
        ret[i] = (*this)[i] + in[i];
        }
        ret.rot_ = a.rot_ * rot_;
        return ret;
    }
    PoseState operator-(const PoseState& a) const
    {
        PoseState in = a;
        PoseState ret;
        for (size_t i = 0; i < 13; i++)
        {
        if (3 <= i && i <= 6)
            continue;
        ret[i] = (*this)[i] - in[i];
        }
        ret.rot_ = a.rot_.inv() * rot_;
        return ret;
    }

    PoseState operator*(const float num) const
    {
        PoseState ret;
        for (size_t i = 0; i < 13; i++)
        {
            ret[i] = (*this)[i] * num;
        }
        return ret;
    }
    
    PoseState operator/(const float num) const
    {
        PoseState ret;
        for (size_t i = 0; i < 13; i++)
        {
            ret[i] = (*this)[i] / num;
        }
        return ret;
    }

    Vec3 pose_;
    Quat rot_;

 
    bool diff_; // has quaternion rpy vec? 
};

#endif