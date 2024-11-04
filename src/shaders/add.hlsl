[[vk::binding(0, 0)]] RWStructuredBuffer<int> InBuffer1;
[[vk::binding(1, 0)]] RWStructuredBuffer<int> InBuffer2;

// Binding 1 in descriptor set 0
[[vk::binding(2, 0)]] RWStructuredBuffer<int> OutBuffer;

[numthreads(1, 1, 1)] void Main(uint3 DTid
                                : SV_DispatchThreadID) {
  OutBuffer[DTid.x] = InBuffer1[DTid.x] + InBuffer2[DTid.x];
}
