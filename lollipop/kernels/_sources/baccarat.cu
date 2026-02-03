/*
 *  Monte Carlo Baccarat simulation — full 8-deck shoe.
 *
 *  Each thread simulates one complete shoe (416 cards).  Cards are
 *  tracked by value counts (not a shuffled array) and drawn via
 *  weighted random selection, equivalent to Fisher-Yates but without
 *  materialising the deck.
 *
 *  Hands are dealt following standard Baccarat third-card rules until
 *  the cut card is reached (~52 cards remaining).  Results per shoe
 *  are accumulated locally, then added to global counters via atomicAdd.
 *
 *  Card values: 0 = {10,J,Q,K} (128 cards), 1-9 = face value (32 each).
 *
 *  Pair tracking (every two consecutive hands):
 *      repeat = same outcome twice  (PP, BB, TT)
 *      change = different outcomes   (PB, BP, PT, TP, BT, TB)
 *
 *  Parameters:
 *      results   — uint32[6]: [player, banker, tie, hands, repeats, changes]
 *      num_shoes — number of shoes to simulate (one per thread)
 *      seed      — RNG seed
 *
 *  Launch: block=(256,), grid=((num_shoes+255)/256,)
 */
__device__ __forceinline__
static int draw_card(unsigned char* count, int& cards_left, unsigned int& rng) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    int r = (int)(rng % (unsigned int)cards_left);
    int val = 0;
    while (val < 9) {
        r -= count[val];
        if (r < 0) break;
        val++;
    }
    count[val]--;
    cards_left--;
    return val;
}

extern "C" __global__
void baccarat(unsigned int* results, int num_shoes, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_shoes) return;

    /* Per-thread xorshift32 RNG */
    unsigned int rng = (unsigned int)tid * 1099087573u + seed;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

    /* Per-value card counts for an 8-deck shoe (416 cards).
       Value 0 covers 10/J/Q/K (4 ranks × 32 = 128 cards).
       Values 1-9 each have 1 rank × 32 = 32 cards. */
    unsigned char count[10];
    count[0] = 128;
    for (int i = 1; i < 10; i++) count[i] = 32;
    int cards_left = 416;

    /* Deal hands until cut card (~1 deck / 52 cards from end) */
    unsigned int lp = 0, lb = 0, lt = 0, lh = 0;
    unsigned int l_repeat = 0, l_change = 0;
    int prev = -1; /* previous outcome: 0=player, 1=banker, 2=tie, -1=none */

    while (cards_left > 52) {
        int p1 = draw_card(count, cards_left, rng);
        int p2 = draw_card(count, cards_left, rng);
        int b1 = draw_card(count, cards_left, rng);
        int b2 = draw_card(count, cards_left, rng);

        int ptotal = (p1 + p2) % 10;
        int btotal = (b1 + b2) % 10;
        int p3 = -1; /* -1 means no third card */

        /* Check for naturals (8 or 9) */
        if (ptotal < 8 && btotal < 8) {

            /* Player draws third card if total 0-5 */
            if (ptotal <= 5) {
                p3 = draw_card(count, cards_left, rng);
                ptotal = (ptotal + p3) % 10;
            }

            /* Banker third-card rules */
            if (p3 == -1) {
                /* Player stood: banker draws on 0-5 */
                if (btotal <= 5) {
                    int b3 = draw_card(count, cards_left, rng);
                    btotal = (btotal + b3) % 10;
                }
            } else {
                /* Player drew: banker decision depends on own total and player's third card */
                int draw = 0;
                if      (btotal <= 2) draw = 1;
                else if (btotal == 3) draw = (p3 != 8);
                else if (btotal == 4) draw = (p3 >= 2 && p3 <= 7);
                else if (btotal == 5) draw = (p3 >= 4 && p3 <= 7);
                else if (btotal == 6) draw = (p3 == 6 || p3 == 7);
                /* btotal 7: stand */

                if (draw) {
                    int b3 = draw_card(count, cards_left, rng);
                    btotal = (btotal + b3) % 10;
                }
            }
        }

        /* Record outcome locally */
        int outcome;
        if      (ptotal > btotal) { lp++; outcome = 0; }
        else if (btotal > ptotal) { lb++; outcome = 1; }
        else                      { lt++; outcome = 2; }
        lh++;

        /* Track repeats and changes between consecutive hands */
        if (prev >= 0) {
            if (outcome == prev) l_repeat++;
            else                 l_change++;
        }
        prev = outcome;
    }

    /* Accumulate per-shoe results */
    atomicAdd(&results[0], lp);        /* player wins   */
    atomicAdd(&results[1], lb);        /* banker wins   */
    atomicAdd(&results[2], lt);        /* ties          */
    atomicAdd(&results[3], lh);        /* total hands   */
    atomicAdd(&results[4], l_repeat);  /* repeat pairs  */
    atomicAdd(&results[5], l_change);  /* change pairs  */
}
