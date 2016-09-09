// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

namespace phe
{

inline Ray::Ray(const vec3& originIn, const vec3& dirIn) noexcept : origin(originIn)
{
	setDir(dirIn);
}

inline void Ray::setOrigin(const vec3& originIn) noexcept
{
	origin = originIn;
}

inline void Ray::setDir(const vec3& dirIn) noexcept
{
	dir = dirIn;
	invDir = vec3(1.0f) / dirIn;
}

} // namespace phe
