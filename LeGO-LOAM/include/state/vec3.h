#ifndef VEC3_H
#define VEC3_H

#include <algorithm>
#include <cmath>

class Vec3 
{
    public:
    inline constexpr Vec3(const float x, const float y, const float z)
    : x_(x), y_(y), z_(z) {}
    
    inline constexpr Vec3(): x_(0), y_(0), z_(0) {}
    
    inline float& operator[](const size_t i)
  {
    switch (i)
    {
      case 0: return x_;
      case 1: return y_;
      case 2: return z_;
      default:
        break;
    }
    return x_;
  }
  inline float operator[](const size_t i) const
  {
    switch (i)
    {
      case 0: return x_;
      case 1: return y_;
      case 2: return z_;
      default:
        break;
    }
    return x_;
  }
    inline constexpr bool operator==(const Vec3& q) const
    {
        return x_ == q.x_ && y_ == q.y_ && z_ == q.z_;
    }
    inline constexpr bool operator!=(const Vec3& q) const
    {
        return !operator==(q);
    }
    inline constexpr Vec3 operator+(const Vec3& q) const
    {
        return Vec3(x_ + q.x_, y_ + q.y_, z_ + q.z_);
    }
    inline constexpr Vec3 operator-(const Vec3& q) const
    {
        return Vec3(x_ - q.x_, y_ - q.y_, z_ - q.z_);
    }
    inline constexpr Vec3 operator-() const
    {
        return Vec3(-x_, -y_, -z_);
    }
    inline constexpr Vec3 operator*(const float s) const
    {
        return Vec3(x_ * s, y_ * s, z_ * s);
    }
    inline constexpr Vec3 operator/(const float s) const
    {
        return Vec3(x_ / s, y_ / s, z_ / s);
    }
    inline Vec3& operator+=(const Vec3& q)
    {
        *this = *this + q;
        return *this;
    }
    inline Vec3& operator-=(const Vec3& q)
    {
        *this = *this - q;
        return *this;
    }
    inline Vec3& operator*=(const float& s)
    {
        *this = *this * s;
        return *this;
    }
    inline Vec3& operator/=(const float& s)
    {
        *this = *this / s;
        return *this;
    }
    inline constexpr float dot(const Vec3& q) const
    {
        return x_ * q.x_ + y_ * q.y_ + z_ * q.z_;
    }
    inline constexpr Vec3 cross(const Vec3& q) const
    {
        return Vec3(y_ * q.z_ - z_ * q.y_,
                    z_ * q.x_ - x_ * q.z_,
                    x_ * q.y_ - y_ * q.x_);
    }
    inline constexpr Vec3 times(const Vec3& q) const
    {
        return Vec3(x_ * q.x_, y_ * q.y_, z_ * q.z_);
    }
    inline float norm() const
    {
        return std::sqrt(dot(*this));
    }
    inline Vec3 normalized() const
    {
        return *this / norm();
    }

    public:
        float x_, y_, z_;    
};


#endif
