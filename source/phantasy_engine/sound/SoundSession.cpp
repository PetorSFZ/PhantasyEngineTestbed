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

#include "phantasy_engine/sound/SoundSession.hpp"

#include <algorithm>

#include "sfz/Assert.hpp"

namespace sfz {

namespace sdl {

// SoundSession: Constructors & destructors
// ------------------------------------------------------------------------------------------------

SoundSession::SoundSession(SoundSession&& other) noexcept
{
	std::swap(this->mActive, other.mActive);
}

SoundSession& SoundSession::operator= (SoundSession&& other) noexcept
{
	std::swap(this->mActive, other.mActive);
	return *this;
}

SoundSession::SoundSession(initializer_list<MixInitFlags> mixInitFlags) noexcept
:
	mActive{true}
{
	// Initialize SDL2_mixer
	int mixInitFlag = 0;
	for (MixInitFlags tempFlag : mixInitFlags) {
		mixInitFlag = mixInitFlag | static_cast<int>(tempFlag);
	}
	int mixInitted = Mix_Init(mixInitFlag);
	if ((mixInitted & mixInitFlag) != mixInitFlag) {
		sfz::error("Mix_Init() failed: %s", Mix_GetError());
	}

	// Open 44.1KHz, signed 16bit, system byte order, stereo audio, using 1024 byte chunks
	if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 1024) < 0) {
		sfz::error("Mix_OpenAudio() failed: %s", Mix_GetError());
	}

	// Allocate mixing channels
	Mix_AllocateChannels(64);
}
	
SoundSession::~SoundSession() noexcept
{
	if (mActive) {
		// Cleanup SDL2_mixer
		Mix_AllocateChannels(0); // Deallocate mixing channels
		Mix_CloseAudio();
		while (Mix_Init(0)) Mix_Quit();
	}
	mActive = false;
}

} // namespace sdl
} // namespace sfz
