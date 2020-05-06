__kernel void generate_mask(__global Fr *_,
                            __global Fr *output,
                            Fr replica_id,
                            uint window_index) {

  uint node_index = get_global_id(0); // Nodes are processed in parallel
  uint layer_index = 1; // Mask layer is always layer 1 (Or 0?)
  sha256_state state = sha256(hash_prefix(layer_index, node_index, window_index, replica_id));
  output[node_index] = sha256_state_to_Fr(state);
}
