#ifndef QUAT_H
#define QUAT_H

#include <algorithm>
#include <cmath>
#include <state/vec3.h>



class Quat 
{
    public:
    inline constexpr Quat(const float& x, const float& y, const float& z, const float &w)
    : x_(x), y_(y), z_(z), w_(w){}

    inline constexpr Quat() : x_(0), y_(0), z_(0), w_(1) {}
    
    inline Quat(const Vec3& axis, const float& ang)
    {
        setAxisAng(axis, ang);
    }
    inline Quat(const Vec3& forward, const Vec3& up_raw)
    {
        const Vec3 xv = forward.normalized();
        const Vec3 yv = up_raw.cross(xv).normalized();
        const Vec3 zv = xv.cross(yv).normalized();

        w_ = std::sqrt(std::max(0.0, 1.0 + xv.x_ + yv.y_ + zv.z_)) / 2.0;
        x_ = std::sqrt(std::max(0.0, 1.0 + xv.x_ - yv.y_ - zv.z_)) / 2.0;
        y_ = std::sqrt(std::max(0.0, 1.0 - xv.x_ + yv.y_ - zv.z_)) / 2.0;
        z_ = std::sqrt(std::max(0.0, 1.0 - xv.x_ - yv.y_ + zv.z_)) / 2.0;
        if (zv.y_ - yv.z_ > 0)
        x_ = -x_;
        if (xv.z_ - zv.x_ > 0)
        y_ = -y_;
        if (yv.x_ - xv.y_ > 0)
        z_ = -z_;
    }
    inline explicit Quat(const Vec3& rpy)
    {
        setRPY(rpy);
    }
    
    inline constexpr float dot(const Quat& q) const
    {
        return x_ * q.x_ + y_ * q.y_ + z_ * q.z_ + w_ * q.w_;
    }
    inline float norm() const
    {
        return std::sqrt(dot(*this));
    }
    inline constexpr bool operator==(const Quat& q) const
    {
        return x_ == q.x_ && y_ == q.y_ && z_ == q.z_ && w_ == q.w_;
    }
    inline constexpr bool operator!=(const Quat& q) const
    {
        return !operator==(q);
    }
    inline Quat operator+(const Quat& q) const
    {
        return Quat(x_ + q.x_, y_ + q.y_, z_ + q.z_, w_ + q.w_);
    }
    inline Quat operator+=(const Quat& q)
    {
        x_ += q.x_;
        y_ += q.y_;
        z_ += q.z_;
        w_ += q.w_;
        return *this;
    }
    inline Quat operator-(const Quat& q) const
    {
        return Quat(x_ - q.x_, y_ - q.y_, z_ - q.z_, w_ - q.w_);
    }
    inline Quat operator-=(const Quat& q)
    {
        x_ -= q.x_;
        y_ -= q.y_;
        z_ -= q.z_;
        w_ -= q.w_;
        return *this;
    }
    inline constexpr Quat operator-() const
    {
        return Quat(-x_, -y_, -z_, -w_);
    }
    inline constexpr Quat operator*(const Quat& q) const
    {
        return Quat(
            w_ * q.x_ + x_ * q.w_ + y_ * q.z_ - z_ * q.y_,
            w_ * q.y_ + y_ * q.w_ + z_ * q.x_ - x_ * q.z_,
            w_ * q.z_ + z_ * q.w_ + x_ * q.y_ - y_ * q.x_,
            w_ * q.w_ - x_ * q.x_ - y_ * q.y_ - z_ * q.z_);
    }
    inline Vec3 operator*(const Vec3& v) const
    {
        const Quat ret = *this * Quat(v.x_, v.y_, v.z_, 0.0) * conj();
        return Vec3(ret.x_, ret.y_, ret.z_);
    }
    inline constexpr Quat operator*(const float& s) const
    {
        return Quat(x_ * s, y_ * s, z_ * s, w_ * s);
    }
    inline constexpr Quat operator/(const float& s) const
    {
        return operator*(1.0 / s);
    }
    inline Quat operator*=(const float& s)
    {
        x_ *= s;
        y_ *= s;
        z_ *= s;
        w_ *= s;
        return *this;
    }
    inline float operator[](const int& i)
    {
        if (i == 0) { return x_; }
        else if (i == 1) { return y_; }
        else if (i == 2) { return z_; }
        else if (i == 3) { return w_; }
        else{ return 100000;}
        
    }
    inline Quat operator/=(const float& s)
    {
        x_ /= s;
        y_ /= s;
        z_ /= s;
        w_ /= s;
        return *this;
    }

    inline Quat weighted(const float& s) const
    {
        Vec3 axis;
        float ang;
        getAxisAng(axis, ang);
        return Quat(axis, ang * s);
    }
    inline Quat normalized() const
    {
        return (*this) / norm();
    }
    inline void normalize()
    {
        *this = normalized();
    }
    inline constexpr Quat conj() const
    {
        return Quat(-x_, -y_, -z_, w_);
    }
    inline constexpr Quat inv() const
    {
        return conj() / dot(*this);
    }
    inline constexpr Vec3 getRPY() const
    {
        const float ysq = y_ * y_;
        const float t0 = -2.0 * (ysq + z_ * z_) + 1.0;
        const float t1 = +2.0 * (x_ * y_ + w_ * z_);
        const float t2 = std::max(-1.0, std::min(1.0, -2.0 * (x_ * z_ - w_ * y_)));
        const float t3 = +2.0 * (y_ * z_ + w_ * x_);
        const float t4 = -2.0 * (x_ * x_ + ysq) + 1.0;

        return Vec3(std::atan2(t3, t4), std::asin(t2), std::atan2(t1, t0));
    }
    inline void setRPY(const Vec3& rpy)
    {
        const float t2 = std::cos(rpy.x_ / 2);
        const float t3 = std::sin(rpy.x_ / 2);
        const float t4 = std::cos(rpy.y_ / 2);
        const float t5 = std::sin(rpy.y_ / 2);
        const float t0 = std::cos(rpy.z_ / 2);
        const float t1 = std::sin(rpy.z_ / 2);

        x_ = t0 * t3 * t4 - t1 * t2 * t5;
        y_ = t0 * t2 * t5 + t1 * t3 * t4;
        z_ = t1 * t2 * t4 - t0 * t3 * t5;
        w_ = t0 * t2 * t4 + t1 * t3 * t5;
    }
    inline void setAxisAng(const Vec3& axis, const float& ang)
    {
        const Vec3 a = axis / axis.norm();
        const float s = std::sin(ang / 2);
        x_ = a.x_ * s;
        y_ = a.y_ * s;
        z_ = a.z_ * s;
        w_ = std::cos(ang / 2);
        normalize();
    }
    inline void getAxisAng(Vec3& axis, float& ang) const
    {
        if (fabs(w_) >= 1.0 - 0.000001)
        {
            ang = 0.0;
            axis = Vec3(0.0, 0.0, 1.0);
            return;
        }
        ang = std::acos(w_) * 2.0;
        if (ang > M_PI)
            ang -= 2.0 * M_PI;
        const float wsq = 1.0 - w_ * w_;
        axis = Vec3(x_, y_, z_) / std::sqrt(wsq);
    }
    inline void rotateAxis(const Quat& r)
    {
        Vec3 axis;
        float ang;
        getAxisAng(axis, ang);
        setAxisAng(r * axis, ang);
    }
    public:
        float x_, y_, z_, w_;
};

#endif