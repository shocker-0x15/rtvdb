#if defined(__INTELLISENSE__)
#   define ARG_IN(Type, var) const Type &var
#   define ARG_OUT(Type, var) Type &var
#   define ARG_INOUT(Type, var) Type &var
#   define LOOP_UNROLL
#   define LOOP_LOOP
#   define CONST_BUFFER struct
#   define REGISTER(reg)

typedef int int32_t;
typedef unsigned int uint32_t;

struct int2 {
    int32_t x;
    int32_t y;
};
struct int3 {
    int32_t x;
    int32_t y;
    int32_t z;
};
struct int4 {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
};

struct uint2 {
    uint32_t x;
    uint32_t y;
};
struct uint3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};
struct uint4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

struct float2 {
    float x;
    float y;
};
struct float3 {
    float x;
    float y;
    float z;
};
struct float4 {
    float x;
    float y;
    float z;
    float w;
};

template <typename T>
struct ConstantBuffer : public T {};

struct RaytracingAccelerationStructure;

struct BuiltInTriangleIntersectionAttributes {
  float2 barycentrics;
};

template <typename T>
struct StructuredBuffer;
template <typename T>
struct RWStructuredBuffer;

template <typename T>
struct Texture2D;
template <typename T>
struct RWTexture2D;
#else
#   define ARG_IN(Type, var) in Type var
#   define ARG_OUT(Type, var) out Type var
#   define ARG_INOUT(Type, var) inout Type var
#   define LOOP_UNROLL [unroll]
#   define LOOP_LOOP [loop]
#   define CONST_BUFFER cbuffer
#   define REGISTER(reg) : register(reg)
#endif

struct PickResult
{
    uint32_t primitive_kind;
    uint32_t primitive_index;
    float distance;
    uint32_t hit;
};

struct GeometryMetadata
{
    uint32_t primitive_base;
    uint32_t index_offset;
    uint32_t primitive_offset;
    uint32_t primitive_count;
};

struct ViewerConstants
{
    float4 origin;
    float4 forward;
    float4 right;
    float4 up;
    float4 scene_bounds_min;
    float4 scene_bounds_max;
    uint4 size_and_mode;
    float4 projection_from;
    float4 projection_to;
    uint4 projection_modes;
    float4 blend_and_jitter;
    uint4 pick_and_flags;
    uint4 pick_params;
    float4 render_scale;
};

struct PointPrimitive
{
    float4 position_radius;
    float4 color;
};

struct LinePrimitive
{
    float4 a_radius;
    float4 b_pad;
    float4 color;
    uint32_t flags;
    float pad0;
    float pad1;
    float pad2;
};

struct ProceduralAttributes
{
    float2 uv;
};

struct Payload
{
    float4 color;
    float hit_t;
    uint32_t hit;
    uint32_t primitive_kind;
    uint32_t primitive_index;
};

static const float kAlphaOpaqueThreshold = 0.999;
static const float kRayTMinFallback = 1.0e-6;
static const float kRayTMax = 10000.0;
static const float kHitAdvanceBiasFallback = 1.0e-6;
static const uint32_t kMaxTransparencyLayers = 16;
static const uint32_t kMaxInstanceGeometryCount = 4;
static const uint32_t kPointPrimitiveSeedBase = 1000000;
static const uint32_t kLinePrimitiveSeedBase = 2000000;
static const uint32_t kLineFlagFixedColor = 1u << 0;
static const uint32_t kLineFlagNonPickable = 1u << 1;

static const uint32_t kProjectionPerspective = 0;
static const uint32_t kProjectionFisheye = 1;
static const uint32_t kProjectionOrthographic = 2;
