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
