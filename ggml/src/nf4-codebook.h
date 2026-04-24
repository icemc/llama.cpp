#pragma once

// NF4 codebook: 16 midpoint quantiles of N(0,1) from QLoRA (Dettmers 2023).
// Constructed as 8 negative quantiles + 0.0 + 7 positive quantiles.
// Index 7 = exactly 0.0. Use index 7 for zero-padding (packed byte 0x77).
static const float NF4_CODEBOOK[16] = {
    -1.0000000f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.0910500f,  0.0000000f,
     0.0795803f,  0.1609302f,  0.2461123f,  0.3379152f,
     0.4407098f,  0.5626170f,  0.7229568f,  1.0000000f,
};

#define NF4_ZERO_IDX 7  // index of the 0.0 codeword
