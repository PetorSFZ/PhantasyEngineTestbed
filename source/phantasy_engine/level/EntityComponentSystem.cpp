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

	// Keeps track of the next index to allocate, and any eventual unallocated slots
	uint32_t nextFreeEntity;
	DynArray<uint32_t> freeSlots;

	// Bitmasks used to determine what components an entity has.
	DynArray<ComponentMask> masks;

	// The components
	DynArray<ComponentType> components;

	EntityComponentSystemImpl(uint32_t maxNumEntities) noexcept
	{
		this->maxNumEntities = maxNumEntities;

		// Initialize number of entities
		this->nextFreeEntity = 0;
		freeSlots.setCapacity(1024);

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

EntityComponentSystem::EntityComponentSystem(uint32_t maxNumEntities) noexcept
{
	this->create(maxNumEntities);
}

EntityComponentSystem::~EntityComponentSystem() noexcept
{
	this->destroy();
}

// EntityComponentSystem: State methods
// ------------------------------------------------------------------------------------------------

void EntityComponentSystem::create(uint32_t maxNumEntities) noexcept
{
	this->destroy();
	mImpl = sfz_new<EntityComponentSystemImpl>(maxNumEntities);
}

void EntityComponentSystem::destroy() noexcept
{
	sfz_delete(mImpl);
	mImpl = nullptr;
}

// EntityComponentSystem: Entity methods
// ------------------------------------------------------------------------------------------------

uint32_t EntityComponentSystem::createEntity() noexcept
{
	sfz_assert_debug(this->currentNumEntities() < this->maxNumEntities());

	// Get free entity (prefer from free list)
	uint32_t entity;
	if (mImpl->freeSlots.size() > 0u) {
		entity = mImpl->freeSlots.last();
		mImpl->freeSlots.removeLast();
	}
	else {
		entity = mImpl->nextFreeEntity;
		mImpl->nextFreeEntity += 1u;
	}

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

	// Add entity to free list if it is not the highest entity index
	if ((entity + 1) == mImpl->nextFreeEntity) {
		mImpl->nextFreeEntity = entity;
	}
	else {
		mImpl->freeSlots.add(entity);
	}
}

const ComponentMask& EntityComponentSystem::componentMask(uint32_t entity) const noexcept
{
	return mImpl->masks[entity];
}

const ComponentMask* EntityComponentSystem::componentMaskArrayPtr() const noexcept
{
	return mImpl->masks.data();
}

uint32_t EntityComponentSystem::maxNumEntities() const noexcept
{
	return mImpl->maxNumEntities;
}

uint32_t EntityComponentSystem::currentNumEntities() const noexcept
{
	return mImpl->nextFreeEntity - mImpl->freeSlots.size();
}

uint32_t EntityComponentSystem::entityIndexUpperBound() const noexcept
{
	return mImpl->nextFreeEntity;
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
	tmp.numComponents = 0u;

	// Add component type and return index for it
	mImpl->components.add(std::move(tmp));
	return mImpl->components.size() - 1u;
}

uint32_t EntityComponentSystem::currentNumComponentTypes() const noexcept
{
	return mImpl->components.size();
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
	for (uint32_t i = 0; i < this->entityIndexUpperBound(); i++) {
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
