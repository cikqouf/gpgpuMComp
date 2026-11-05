[[vk::binding(0, 0)]]
RWStructuredBuffer<float> buf;

// todo: float only, prepare to support other Types...
[[vk::push_constant]]
float a;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    buf[i] = buf[i] * a;
}
