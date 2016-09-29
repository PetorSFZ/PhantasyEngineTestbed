// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/geometry/AABB.hpp>
#include <sfz/geometry/OBB.hpp>
#include <sfz/geometry/Plane.hpp>
#include <sfz/geometry/Sphere.hpp>
#include <sfz/math/Vector.hpp>

// Stupid hack for stupid near/far macros (windows.h)
#undef near
#undef far

namespace phe {

using sfz::AABB;
using sfz::OBB;
using sfz::Plane;
using sfz::Sphere;
using sfz::mat4;
using sfz::vec2;
using sfz::vec2i;
using sfz::vec3;

class ViewFrustum final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	ViewFrustum() noexcept = default;
	ViewFrustum(const ViewFrustum&) noexcept = default;
	ViewFrustum& operator= (const ViewFrustum&) noexcept = default;

	ViewFrustum(vec3 position, vec3 direction, vec3 up, float verticalFovDeg, float aspect,
	            float near, float far) noexcept;
	
	// Public methods
	// --------------------------------------------------------------------------------------------

	bool isVisible(const AABB& aabb) const noexcept;
	bool isVisible(const OBB& obb) const noexcept;
	bool isVisible(const Sphere& sphere) const noexcept;
	bool isVisible(const ViewFrustum& viewFrustum) const noexcept;

	// Getters
	// --------------------------------------------------------------------------------------------

	inline vec3 pos() const noexcept { return mPos; }
	inline vec3 dir() const noexcept { return mDir; }
	inline vec3 up() const noexcept { return mUp; }
	inline float verticalFov() const noexcept { return mVerticalFovDeg; }
	inline float aspectRatio() const noexcept { return mAspectRatio; }
	inline float near() const noexcept { return mNear; }
	inline float far() const noexcept { return mFar; }
	mat4 viewMatrix() const noexcept;
	mat4 projMatrix(vec2i resolution) const noexcept;

	// Setters
	// --------------------------------------------------------------------------------------------

	void setPos(vec3 position) noexcept;
	void setVerticalFov(float verticalFovDeg) noexcept;
	void setAspectRatio(float aspect) noexcept;

	void setDir(vec3 direction, vec3 up) noexcept;
	void setClipDist(float near, float far) noexcept;
	void set(vec3 position, vec3 direction, vec3 up, float verticalFovDeg, float aspect, float near,
	         float far) noexcept;
	void setPixelOffset(vec2 offset) noexcept;

private:
	// Private methods
	// --------------------------------------------------------------------------------------------

	void update() noexcept;
	void updatePlanes() noexcept;

	// Private members
	// --------------------------------------------------------------------------------------------

	vec3 mPos, mDir, mUp;
	float mVerticalFovDeg, mAspectRatio, mNear, mFar;
	vec2 mPixelOffset = vec2(0.0f);
	Plane mNearPlane, mFarPlane, mUpPlane, mDownPlane, mLeftPlane, mRightPlane;
};

} // namespace phe
