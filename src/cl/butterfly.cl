__kernel void generate_butterfly(__global Fr *input,
                                 __global Fr *output,
                                 uint layer_index) {

  uint v = get_global_id(0); // Nodes are processed in parallel

  uint factor = 1 << (LOG2_DEGREE_BUTTERFLY * (NUM_LAYERS - layer_index));

  sha256_state state = sha256_INIT;

  for(uint i = 0; i < DEGREE_BUTTERFLY / 2; i++) {
    uint i_1 = i * 2;
    uint i_2 = i * 2 + 1;

    uint parent_1 = (v + i_1 * factor) & MODULO_N_MASK;
    uint parent_2 = (v + i_2 * factor) & MODULO_N_MASK;

    state = sha256_update(state, Fr_to_sha256_data(input[parent_1], input[parent_2]));
  }

  state = sha256_finish(state, DEGREE_BUTTERFLY / 2);

  output[v] = sha256_state_to_Fr(state);
}
