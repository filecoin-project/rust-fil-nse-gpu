__kernel void generate_mask(__global Fr *output,
                            replica_id id,
                            uint window_index) {

  uint node_index = get_global_id(0); // Nodes are processed in parallel
  uint layer_index = 1; // Mask layer is always layer 1 (Or 0?)
  sha256_domain state = sha256(hash_prefix(layer_index, node_index, window_index, id));
  output[node_index] = sha256_domain_to_Fr(state);
}
