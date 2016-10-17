// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#pragma once

#include <sfz/CUDACallable.hpp>
#include <sfz/math/MathHelpers.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using sfz::vec4;
using sfz::vec4i;

// Material
// ------------------------------------------------------------------------------------------------

struct Material final {

	// Raw material data, should not be accessed directly
	// --------------------------------------------------------------------------------------------

	/// A material is stored as follows:
	///
	/// [albedoTexIndex, roughnessTexIndex, metallicTexIndex, padding]
	/// [albedoValue.red, albedoValue.green, albedoValue.blue, albedoValue.alpha]
	/// [roughnessValue, metallicValue, padding, padding]
	///
	/// If a texture index is -1 then no texture is available.

	vec4i iData = vec4i(-1);
	vec4 fData1 = vec4(0.0f);
	vec4 fData2 = vec4(0.0f);

	// Getters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE int32_t albedoTexIndex() const noexcept { return iData.x; }
	SFZ_CUDA_CALLABLE int32_t roughnessTexIndex() const noexcept { return iData.y; }
	SFZ_CUDA_CALLABLE int32_t metallicTexIndex() const noexcept { return iData.z; }
	
	SFZ_CUDA_CALLABLE vec4 albedoValue() const noexcept { return fData1; }
	SFZ_CUDA_CALLABLE float roughnessValue() const noexcept { return fData2.x; }
	SFZ_CUDA_CALLABLE float metallicValue() const noexcept { return fData2.y; }

	// Setters
	// --------------------------------------------------------------------------------------------

	SFZ_CUDA_CALLABLE void setAlbedoTexIndex(int32_t index) noexcept { iData.x = index; }
	SFZ_CUDA_CALLABLE void setRoughnessTexIndex(int32_t index) noexcept { iData.y = index; }
	SFZ_CUDA_CALLABLE void setMetallicTexIndex(int32_t index) noexcept { iData.z = index; }

	SFZ_CUDA_CALLABLE void setAlbedoValue(const vec4& value) noexcept { fData1 = value; }
	SFZ_CUDA_CALLABLE void setRoughnessValue(float value) noexcept { fData2.x = value; }
	SFZ_CUDA_CALLABLE void setMetallicValue(float value) noexcept { fData2.y = value; }
};

// Material comparison functions
// ------------------------------------------------------------------------------------------------

inline bool approxEqual(const Material& lhs, const Material& rhs) noexcept
{
	return lhs.iData == rhs.iData &&
	       sfz::approxEqual(lhs.fData1, rhs.fData1) &&
	       sfz::approxEqual(lhs.fData2, rhs.fData2);
}


} // namespace phe
