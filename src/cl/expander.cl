__kernel void generate_expander(__global Fr *input,
                                __global Fr *output,
                                uint layer_index) {

  uint node = get_global_id(0); // Nodes are processed in parallel

  bit_stream stream = gen_stream(node); // 1152 Bytes ~ 1KB

  sha256_state state = sha256_INIT;
  sha256_data data = sha256_ZERO;

  for(uint i = 0; i < DEGREE_EXPANDER / 2; i++) {
    Fr x_1 = Fr_ZERO;
    Fr x_2 = Fr_ZERO;

    for(uint j = 0; j < K; j++) {
      uint parent_1 = get_expanded_parent(&stream, (i * 2) + j * DEGREE_EXPANDER);
      uint parent_2 = get_expanded_parent(&stream, (i * 2 + 1) + j * DEGREE_EXPANDER);

      x_1 = Fr_add(x_1, input[parent_1]);
      x_2 = Fr_add(x_2, input[parent_2]);
    }

    state = sha256_update(state, Fr_to_sha256_data(x_1, x_2));
  }

  state = sha256_finish(state, DEGREE_EXPANDER / 2);

  output[node] = sha256_state_to_Fr(state);
}
