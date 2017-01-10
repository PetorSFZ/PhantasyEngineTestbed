// Copyright (c) Peter Hillerström (skipifzero.com, peter@hstroem.se)
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#pragma once

#include <initializer_list>

#include <SDL_mixer.h>

namespace sfz {

namespace sdl {

using std::initializer_list;

// Enums
// ------------------------------------------------------------------------------------------------

/// SDL2_mixer init flags
enum class MixInitFlags {
	FLAC = MIX_INIT_FLAC,
	MOD = MIX_INIT_MOD,
	MP3 = MIX_INIT_MP3,
	OGG = MIX_INIT_OGG
};

// SoundSession class
// ------------------------------------------------------------------------------------------------

/// Initializes SDL2_mixer upon construction and cleans up upon destruction. This object must be
/// kept alive as long as SDL2_mixer is used. An ordinary sfz::sdl::Session needs to be alive
/// during the whole lifetime of a SoundSession.
class SoundSession final {
public:
	// Constructors & destructors
	// --------------------------------------------------------------------------------------------

	// Copying not allowed
	SoundSession(const SoundSession&) = delete;
	SoundSession& operator= (const SoundSession&) = delete;

	SoundSession() noexcept = default;
	SoundSession(SoundSession&& other) noexcept;
	SoundSession& operator= (SoundSession&& other) noexcept;

	/// Initializes SDL2_mixer with the specified flags
	/// SDL_mixer will open audio with: 44.1KHz, signed 16bit, system byte order, stereo audio,
	/// 1024 byte chunks. Additionally 64 mixing channels will be allocated.
	explicit SoundSession(initializer_list<MixInitFlags> mixInitFlags) noexcept;
	~SoundSession() noexcept;

private:
	// Private members
	// --------------------------------------------------------------------------------------------

	bool mActive = false;
};

} // namespace sdl
} // namespace sfz
