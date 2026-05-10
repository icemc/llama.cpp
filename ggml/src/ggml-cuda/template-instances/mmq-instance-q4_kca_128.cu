// MMQ template instantiation for Q4_KCA_128.
// Iterates at K-quant group granularity (QK_KCA_G=256); load_tiles navigates
// the two-level super-block structure (8 groups × 256 weights = 2048 weights).

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_Q4_KCA_128);
