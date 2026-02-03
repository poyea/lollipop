/*
 *  FFT butterfly — one stage of the Cooley-Tukey radix-2 DIT FFT.
 *
 *  The full FFT is computed by calling this kernel log2(n) times,
 *  once per stage (stage = 0, 1, ..., log2(n)-1).
 *
 *  Each thread performs one butterfly operation:
 *    twiddle = exp(-2*pi*i * pos / size)
 *    (re[i], im[i])  = (re[i] + tw*re[j],  im[i] + tw*im[j])
 *    (re[j], im[j])  = (re[i] - tw*re[j],  im[i] - tw*im[j])
 *
 *  Note: this is a decimation-in-time (DIT) FFT without bit-reversal
 *  permutation.  Input should be in bit-reversed order, or use only
 *  with inputs that are invariant to bit-reversal (e.g. DC, impulse at 0).
 *
 *  Parameters:
 *      re, im — real and imaginary parts (n float32 each, in-place)
 *      n      — transform size (must be power of 2)
 *      stage  — current butterfly stage (0 to log2(n)-1)
 *
 *  Launch: block=(256,), grid=((n/2+255)/256,)
 */
extern "C" __global__
void fft_butterfly(float* re, float* im, int n, int stage) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_size = 1 << stage;
    int size = half_size << 1;
    int group = tid / half_size;
    int pos = tid % half_size;

    int i = group * size + pos;
    int j = i + half_size;
    if (j >= n) return;

    float angle = -2.0f * 3.14159265358979f * pos / size;
    float wr = cosf(angle);
    float wi = sinf(angle);

    float tr = re[j] * wr - im[j] * wi;
    float ti = re[j] * wi + im[j] * wr;

    re[j] = re[i] - tr;
    im[j] = im[i] - ti;
    re[i] = re[i] + tr;
    im[i] = im[i] + ti;
}
