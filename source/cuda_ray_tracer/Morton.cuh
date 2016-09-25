// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

// Morton encoding and decoding functions courtesy of Fabian “ryg” Giesen
// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// I would guess that it's okay to use, but no proper license is given.

#pragma once

#include <cstdint>

#include <sfz/CUDACallable.hpp>
#include <sfz/math/Vector.hpp>

namespace phe {

using std::uint32_t;
using sfz::vec2i;
using sfz::vec2u;
using sfz::vec3i;
using sfz::vec3u;

// Morton internals
// ------------------------------------------------------------------------------------------------

// "Insert" a 0 bit after each of the 16 low bits of x
SFZ_CUDA_CALLABLE uint32_t mortonPart1By1(uint32_t x) noexcept
{
	x &= 0x0000FFFFu;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x << 8u)) & 0x00FF00FFu; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x << 4u)) & 0x0F0F0F0Fu; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x << 2u)) & 0x33333333u; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x << 1u)) & 0x55555555u; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
SFZ_CUDA_CALLABLE uint32_t mortonPart1By2(uint32_t x) noexcept
{
	x &= 0x000003FFu;                   // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16u)) & 0xFF0000FFu; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8u)) & 0x0300F00Fu; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4u)) & 0x030C30C3u; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2u)) & 0x09249249u; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

// Inverse of Part1By1 - "delete" all odd-indexed bits
SFZ_CUDA_CALLABLE uint32_t mortonCompact1By1(uint32_t x) noexcept
{
	x &= 0x55555555u;                   // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >>  1u)) & 0x33333333u; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >>  2u)) & 0x0F0F0F0Fu; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >>  4u)) & 0x00FF00FFu; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >>  8u)) & 0x0000FFFFu; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
SFZ_CUDA_CALLABLE uint32_t mortonCompact1By2(uint32_t x) noexcept
{
	x &= 0x09249249u;                   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >>  2u)) & 0x030C30C3u; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >>  4u)) & 0x0300F00Fu; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >>  8u)) & 0xFF0000FFu; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16u)) & 0x000003FFu; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

// Morton (2) encoding
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE uint32_t encodeMorton2(uint32_t x, uint32_t y) noexcept
{
	return (mortonPart1By1(y) << 1u) | mortonPart1By1(x);
}

SFZ_CUDA_CALLABLE uint32_t encodeMorton2(vec2u loc) noexcept
{
	return encodeMorton2(loc.x, loc.y);
}

SFZ_CUDA_CALLABLE uint32_t encodeMorton2(vec2i loc) noexcept
{
	return encodeMorton2(static_cast<vec2u>(loc));
}

// Morton (3) encoding
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE uint32_t encodeMorton3(uint32_t x, uint32_t y, uint32_t z) noexcept
{
	return (mortonPart1By2(z) << 2u) + (mortonPart1By2(y) << 1u) + mortonPart1By2(x);
}

SFZ_CUDA_CALLABLE uint32_t encodeMorton3(vec3u loc) noexcept
{
	return encodeMorton3(loc.x, loc.y, loc.z);
}

SFZ_CUDA_CALLABLE uint32_t encodeMorton3(vec3i loc) noexcept
{
	return encodeMorton3(static_cast<vec3u>(loc));
}

// Morton (2) decoding
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE uint32_t decodeMorton2X(uint32_t code) noexcept
{
	return mortonCompact1By1(code);
}

SFZ_CUDA_CALLABLE uint32_t decodeMorton2Y(uint32_t code) noexcept
{
	return mortonCompact1By1(code >> 1);
}

SFZ_CUDA_CALLABLE vec2u decodeMorton2(uint32_t code) noexcept
{
	return vec2u(decodeMorton2X(code), decodeMorton2Y(code));
}

// Morton (3) decoding
// ------------------------------------------------------------------------------------------------

SFZ_CUDA_CALLABLE uint32_t decodeMorton3X(uint32_t code) noexcept
{
	return mortonCompact1By2(code);
}

SFZ_CUDA_CALLABLE uint32_t decodeMorton3Y(uint32_t code) noexcept
{
	return mortonCompact1By2(code >> 1);
}

SFZ_CUDA_CALLABLE uint32_t decodeMorton3Z(uint32_t code) noexcept
{
	return mortonCompact1By2(code >> 2);
}

SFZ_CUDA_CALLABLE vec3u decodeMorton3(uint32_t code) noexcept
{
	return vec3u(decodeMorton3X(code), decodeMorton3Y(code), decodeMorton3Z(code));
}

} // namespace phe
