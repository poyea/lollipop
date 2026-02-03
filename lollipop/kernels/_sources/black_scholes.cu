/*
 *  Black-Scholes European option pricing.
 *
 *  Closed-form solution for European call and put prices:
 *      d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma * sqrt(T))
 *      d2 = d1 - sigma * sqrt(T)
 *      call = S * N(d1) - K * e^(-rT) * N(d2)
 *      put  = K * e^(-rT) * N(-d2) - S * N(-d1)
 *
 *  Uses CUDA's built-in normcdff() for the cumulative normal distribution.
 *  Each thread prices one option contract independently.
 *
 *  Parameters:
 *      spot       — current underlying prices  (n float32)
 *      strike     — strike prices               (n float32)
 *      ttm        — time to maturity in years   (n float32)
 *      rate       — risk-free interest rates     (n float32)
 *      vol        — implied volatilities         (n float32)
 *      call_price — output call prices           (n float32)
 *      put_price  — output put prices            (n float32)
 *      n          — number of option contracts
 *
 *  Launch: block=(256,), grid=((n+255)/256,)
 */
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
