extern "C" __global__
void baccarat(unsigned int* results, int num_hands, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hands) return;

    /* Per-thread xorshift32 RNG */
    unsigned int rng = (unsigned int)tid * 1099087573u + seed;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;

    /* Card value lookup: A=1, 2-9=face, 10/J/Q/K=0 */
    const int card_val[13] = {1,2,3,4,5,6,7,8,9,0,0,0,0};

    #define DRAW_CARD() ( \
        rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5, \
        card_val[rng % 13] \
    )

    /* Deal initial 2 cards each */
    int p1 = DRAW_CARD();
    int p2 = DRAW_CARD();
    int b1 = DRAW_CARD();
    int b2 = DRAW_CARD();

    int player_total = (p1 + p2) % 10;
    int banker_total = (b1 + b2) % 10;

    int p3 = -1; /* -1 means no third card */

    /* Check for naturals (8 or 9) */
    if (player_total < 8 && banker_total < 8) {

        /* Player draws third card if total 0-5 */
        if (player_total <= 5) {
            p3 = DRAW_CARD();
            player_total = (player_total + p3) % 10;
        }

        /* Banker third-card rules */
        if (p3 == -1) {
            /* Player stood: banker draws on 0-5 */
            if (banker_total <= 5) {
                int b3 = DRAW_CARD();
                banker_total = (banker_total + b3) % 10;
            }
        } else {
            /* Player drew: banker decision depends on own total and player's third card */
            int draw = 0;
            if (banker_total <= 2) {
                draw = 1;
            } else if (banker_total == 3) {
                draw = (p3 != 8);
            } else if (banker_total == 4) {
                draw = (p3 >= 2 && p3 <= 7);
            } else if (banker_total == 5) {
                draw = (p3 >= 4 && p3 <= 7);
            } else if (banker_total == 6) {
                draw = (p3 == 6 || p3 == 7);
            }
            /* banker_total 7: stand */

            if (draw) {
                int b3 = DRAW_CARD();
                banker_total = (banker_total + b3) % 10;
            }
        }
    }

    /* Record outcome: 0=player wins, 1=banker wins, 2=tie */
    if (player_total > banker_total) {
        atomicAdd(&results[0], 1);
    } else if (banker_total > player_total) {
        atomicAdd(&results[1], 1);
    } else {
        atomicAdd(&results[2], 1);
    }

    #undef DRAW_CARD
}
