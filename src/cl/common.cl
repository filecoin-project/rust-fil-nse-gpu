#define BITS_PER_BYTE (8)
#define NUM_LAYERS (NUM_EXPANDER_LAYERS + NUM_BUTTERFLY_LAYERS)
#define MODULO_N_MASK (N - 1)

sha256_data hash_prefix(uint layer_index, uint node_index, uint window_index, Fr replica_id) {
  sha256_data data = sha256_ZERO;
  data.vals[0] = layer_index;
  data.vals[1] = node_index;
  data.vals[2] = window_index;
  // TODO: Fill `data[8:]`` with `replica_id`
  return data;
}

sha256_data Fr_to_sha256_data(Fr a, Fr b) {
  sha256_data data;
  for(uint i = 0; i < Fr_LIMBS; i++) {
    data.vals[2 * i] = a.val[i] & 0xffffffff;
    data.vals[2 * i + 1] = a.val[i] >> 32;
    data.vals[2 * (i + Fr_LIMBS)] = b.val[i] & 0xffffffff;
    data.vals[2 * (i + Fr_LIMBS) + 1] = b.val[i] >> 32;
  }
  return data;
}

Fr sha256_state_to_Fr(sha256_state state) {
  Fr f;
  for(uint i = 0; i < Fr_LIMBS; i++)
    f.val[i] = (state.vals[2 * i + 1] << 32) + state.vals[2 * i];
  // Zeroing out last two bits
  f.val[Fr_LIMBS - 1] <<= 2;
  f.val[Fr_LIMBS - 1] >>= 2;
  return f;
}
