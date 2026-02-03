extern "C" __global__
void black_scholes(const float* spot, const float* strike, const float* ttm,
                   const float* rate, const float* vol,
                   float* call_price, float* put_price, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float S = spot[i];
    float K = strike[i];
    float T = ttm[i];
    float r = rate[i];
    float sigma = vol[i];

    float sqrt_T = sqrtf(T);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrt_T);
    float d2 = d1 - sigma * sqrt_T;

    float Nd1 = normcdff(d1);
    float Nd2 = normcdff(d2);

    float discount = expf(-r * T);

    call_price[i] = S * Nd1 - K * discount * Nd2;
    put_price[i] = K * discount * (1.0f - Nd2) - S * (1.0f - Nd1);
}
