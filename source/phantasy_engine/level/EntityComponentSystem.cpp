// See 'LICENSE_PHANTASY_ENGINE' for copyright and contributors.

#include "phantasy_engine/level/EntityComponentSystem.hpp"

#include <algorithm>

#include <immintrin.h> // Intel AVX intrinsics

#include <sfz/Assert.hpp>
#include <sfz/containers/DynArray.hpp>
#include <sfz/memory/New.hpp>

namespace phe {

using namespace sfz;

// ComponentMask: Constructors & destructors
// ------------------------------------------------------------------------------------------------

ComponentMask ComponentMask::empty() noexcept
{
	return fromRawValue(0, 0);
}

ComponentMask ComponentMask::fromType(uint32_t componentType) noexcept
{
	__m128i val;
	if (componentType < 64) {
		uint64_t high = 0;
		uint64_t low = uint64_t(1) << uint64_t(componentType);
		val = _mm_setr_epi64x(low, high);
	}
	else {
		uint64_t high = uint64_t(1) << uint64_t(componentType - 64);
		uint64_t low = 0;
		val = _mm_setr_epi64x(low, high);
	}
	ComponentMask tmp;
	_mm_store_si128((__m128i*)tmp.rawMask, val);
	return tmp;
}

ComponentMask ComponentMask::fromRawValue(uint64_t highBits, uint64_t lowBits) noexcept
{
	__m128i val = _mm_setr_epi64x(lowBits, highBits);
	ComponentMask tmp;
	_mm_store_si128((__m128i*)tmp.rawMask, val);
	return tmp;
}

// ComponentMask: Operators
// ------------------------------------------------------------------------------------------------

bool ComponentMask::operator== (const ComponentMask& other) const noexcept
{
	__m128i thisReg = _mm_load_si128((const __m128i*)this->rawMask);
	__m128i otherReg = _mm_load_si128((const __m128i*)other.rawMask);
	__m128i cmp1 = _mm_cmpeq_epi64(thisReg, otherReg);
	__m128i cmp2 = _mm_and_si128(cmp1, _mm_srli_si128(cmp1, 8));
	return uint64_t(_mm_extract_epi64(cmp2, 0)) != uint64_t(0);
}

bool ComponentMask::operator!= (const ComponentMask& other) const noexcept
{
	return !(*this == other);
}

ComponentMask ComponentMask::operator& (const ComponentMask& other) const noexcept
{
	__m128i thisReg = _mm_load_si128((const __m128i*)this->rawMask);
	__m128i otherReg = _mm_load_si128((const __m128i*)other.rawMask);
	__m128i comb = _mm_and_si128(thisReg, otherReg);

	ComponentMask tmp;
	_mm_store_si128((__m128i*)tmp.rawMask, comb);
	return tmp;
}

ComponentMask ComponentMask::operator| (const ComponentMask& other) const noexcept
{
	__m128i thisReg = _mm_load_si128((const __m128i*)this->rawMask);
	__m128i otherReg = _mm_load_si128((const __m128i*)other.rawMask);
	__m128i disj = _mm_or_si128(thisReg, otherReg);

	ComponentMask tmp;
	_mm_store_si128((__m128i*)tmp.rawMask, disj);
	return tmp;
}

ComponentMask ComponentMask::operator~ () const noexcept
{
	__m128i reg = _mm_load_si128((const __m128i*)this->rawMask);
	__m128i ones; ones = _mm_cmpeq_epi8(ones, ones);
	__m128i negated = _mm_xor_si128(reg, ones);

	ComponentMask tmp;
	_mm_store_si128((__m128i*)tmp.rawMask, negated);
	return tmp;
}

// ComponentMask: Methods
// ------------------------------------------------------------------------------------------------

bool ComponentMask::hasComponentType(uint32_t componentType) const noexcept
{
	return this->fulfills(ComponentMask::fromType(componentType));
}

bool ComponentMask::fulfills(const ComponentMask& constraints) const noexcept
{
	__m128i thisReg = _mm_load_si128((const __m128i*)this->rawMask);
	__m128i constraintsReg = _mm_load_si128((const __m128i*)constraints.rawMask);

	// (this & constraints) == constraints
	__m128i andReg = _mm_and_si128(thisReg, constraintsReg);
	__m128i cmp1 = _mm_cmpeq_epi64(andReg, constraintsReg);
	__m128i cmp2 = _mm_and_si128(cmp1, _mm_srli_si128(cmp1, 8));

	return uint64_t(_mm_extract_epi64(cmp2, 0)) != uint64_t(0);
}

// EntityComponentSystemImpl
// ------------------------------------------------------------------------------------------------

struct ComponentType {
	uint32_t bytesPerComponent;
	uint32_t numComponents;
	DynArray<uint32_t> entityTable;
	DynArray<uint8_t> data;
};

class EntityComponentSystemImpl final {
public:
	// The maximum number of entities allowed
	uint32_t maxNumEntities;

	// All currently free entity indices in this system
	DynArray<uint32_t> freeEntities;

	// Bitmasks used to determine what components an entity has.
	DynArray<ComponentMask> masks;

	// The components
	DynArray<ComponentType> components;

	EntityComponentSystemImpl(uint32_t maxNumEntities) noexcept
	{
		this->maxNumEntities = maxNumEntities;

		// Create and fill freeEntities with all free indices
		freeEntities.setCapacity(maxNumEntities);
		freeEntities.setSize(maxNumEntities);
		for (uint32_t i = 0; i < maxNumEntities; i++) {
			freeEntities[i] = maxNumEntities - i - 1u;
		}

		// Initialize all bitmasks to 0
		masks = DynArray<ComponentMask>(maxNumEntities);
		memset(masks.data(), 0, maxNumEntities * sizeof(ComponentMask));

		// Allocate memory for the components and set "existence component"
		components = DynArray<ComponentType>(1, ECS_MAX_NUM_COMPONENT_TYPES);
		components[0].bytesPerComponent = 0;
		components[0].numComponents = 0;
	}

	~EntityComponentSystemImpl() noexcept
	{

	}
};

// EntityComponentSystem: Constructors & destructors
// ------------------------------------------------------------------------------------------------

EntityComponentSystem::EntityComponentSystem(uint32_t maxNumEntitiesIn) noexcept
{
	mImpl = sfz_new<EntityComponentSystemImpl>(maxNumEntitiesIn);
}

EntityComponentSystem::EntityComponentSystem(EntityComponentSystem&& other) noexcept
{
	this->swap(other);
}

EntityComponentSystem& EntityComponentSystem::operator= (EntityComponentSystem&& other) noexcept
{
	this->swap(other);
	return *this;
}

EntityComponentSystem::~EntityComponentSystem() noexcept
{
	this->destroy();
}

// EntityComponentSystem: State methods & getters
// ------------------------------------------------------------------------------------------------

void EntityComponentSystem::destroy() noexcept
{
	sfz_delete(mImpl);
	mImpl = nullptr;
}

void EntityComponentSystem::swap(EntityComponentSystem& other) noexcept
{
	std::swap(this->mImpl, other.mImpl);
}

uint32_t EntityComponentSystem::maxNumEntities() const noexcept
{
	return mImpl->maxNumEntities;
}

uint32_t EntityComponentSystem::currentNumEntities() const noexcept
{
	return mImpl->maxNumEntities - mImpl->freeEntities.size();
}

uint32_t EntityComponentSystem::currentNumComponentTypes() const noexcept
{
	return mImpl->components.size();
}

// EntityComponentSystem: Entity methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createEntity() noexcept
{
	// Get free entity
	sfz_assert_release(mImpl->freeEntities.size() > 0);
	uint32_t entity = mImpl->freeEntities.last();
	mImpl->freeEntities.removeLast();

	// Sets existence bit
	ComponentMask& mask = mImpl->masks[entity];
	mask = ComponentMask::fromType(0);

	return entity;
}

void EntityComponentSystem::deleteEntity(uint32_t entity) noexcept
{
	sfz_assert_debug(entity < mImpl->maxNumEntities);

	// Retrieve mask
	ComponentMask& mask = mImpl->masks[entity];
	
	// Check if entity does not exist (first bit is reserved for entity existence)
	if (mask == ComponentMask::empty()) {
		printErrorMessage("EntityComponentSystem: Trying to delete entity that does not exist.");
		return;
	}

	// Remove all associated components (skip 0, it is existence flag)
	for (uint32_t i = 1; i < this->currentNumComponentTypes(); i++) {
		if (mask.hasComponentType(i)) {
			this->removeComponent(entity, i);
		}
	}

	// Clear mask
	mask = ComponentMask::empty();

	// Add entity to list of free entities
	mImpl->freeEntities.add(entity);
}

const ComponentMask& EntityComponentSystem::componentMask(uint32_t entity) const noexcept
{
	return mImpl->masks[entity];
}

// EntityComponentSystem: Raw (non-typesafe) component methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createComponentTypeRaw(uint32_t bytesPerComponent) noexcept
{
	// Allocating memory for component type
	ComponentType tmp;
	tmp.bytesPerComponent = bytesPerComponent;
	tmp.entityTable = DynArray<uint32_t>(maxNumEntities(), ~0u, maxNumEntities());
	tmp.data.setCapacity(bytesPerComponent * maxNumEntities());
	tmp.data.setSize(bytesPerComponent * maxNumEntities());

	// Add component type and return index for it
	mImpl->components.add(std::move(tmp));
	return mImpl->components.size() - 1u;
}

void EntityComponentSystem::addComponentRaw(uint32_t entity, uint32_t componentType,
                                            const void* component) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];
	
	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// If entity does not have component it needs to be added
	if (componentLoc == ~0u) {
		componentLoc = compType.numComponents;
		compType.numComponents += 1u;
		compType.entityTable[entity] = componentLoc;
		mImpl->masks[entity] = mImpl->masks[entity] | ComponentMask::fromType(componentType);
	}

	// Copy data
	uint8_t* componentPtr = compType.data.data() + (componentLoc * compType.bytesPerComponent);
	memcpy(componentPtr, component, compType.bytesPerComponent);
}

void EntityComponentSystem::removeComponent(uint32_t entity, uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to delete component that does not exist.");
		return;
	}

	// Remove component
	uint8_t* dstPtr = compType.data.data() + compType.bytesPerComponent * componentLoc;
	uint8_t* srcPtr = compType.data.data() + compType.bytesPerComponent * (componentLoc + 1u);
	uint32_t numBytesToMove = (mImpl->maxNumEntities - componentLoc - 1u) * compType.bytesPerComponent;
	std::memmove(dstPtr, srcPtr, numBytesToMove);
	compType.numComponents -= 1;

	compType.entityTable[entity] = ~0u;
	mImpl->masks[entity] = mImpl->masks[entity] & (~ComponentMask::fromType(componentType));

	// Update all indices larger than removed index
	// TODO: Do something smarter than maxNumEntities
	for (uint32_t i = 0; i < mImpl->maxNumEntities; i++) {
		uint32_t compLoc = compType.entityTable[i];
		if (compLoc != ~0u && compLoc > componentLoc) {
			compType.entityTable[i] = compLoc - 1u;
		}
	}
}

void* EntityComponentSystem::componentArrayPtrRaw(uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].data.data();
}

const void* EntityComponentSystem::componentArrayPtrRaw(uint32_t componentType) const noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].data.data();
}

uint32_t EntityComponentSystem::numComponents(uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	return mImpl->components[componentType].numComponents;
}

void* EntityComponentSystem::getComponentRaw(uint32_t entity, uint32_t componentType) noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to access component that does not exist.");
		return nullptr;
	}

	return compType.data.data() + (componentLoc * compType.bytesPerComponent);
}

const void* EntityComponentSystem::getComponentRaw(uint32_t entity,
                                                   uint32_t componentType) const noexcept
{
	sfz_assert_debug(componentType < mImpl->components.size());
	ComponentType& compType = mImpl->components[componentType];

	// Retrieve component location for given entity
	sfz_assert_debug(entity < mImpl->maxNumEntities);
	uint32_t componentLoc = compType.entityTable[entity];

	// Check if component exists
	if (componentLoc == ~0u) {
		printErrorMessage("EntityComponentSystem: Trying to access component that does not exist.");
		return nullptr;
	}

	return compType.data.data() + (componentLoc * compType.bytesPerComponent);
}

} // namespace phe
