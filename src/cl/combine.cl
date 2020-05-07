__kernel void combine_segment(__global Fr *mask,
                              __global Fr *data,
                              uint offset,
                              uint len,
                              uint is_decode) {
  uint node = get_global_id(0); // Nodes are processed in parallel

  // TODO: Delete this in future, and limit global work size
  if(node < offset || node >= offset + len)
    return;

  if(is_decode)
    data[node] = Fr_sub(data[node], mask[node]);
  else
    data[node] = Fr_add(data[node], mask[node]);
}
