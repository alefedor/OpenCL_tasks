#define SWAP(a, b) { __local double *tmp = a; a = b; b = tmp; }

__kernel void block_prefix_sum(__global double *input, __global double *output, __local double *prev_sum, __local double *nxt_sum, int size) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    if (gid < size)
        prev_sum[lid] = nxt_sum[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint size = 1; size < block_size; size *= 2) {
        if (lid >= size) {
            nxt_sum[lid] = prev_sum[lid] + prev_sum[lid - size];
        } else {
            nxt_sum[lid] = prev_sum[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(prev_sum, nxt_sum);
    }
    if (gid < size)
        output[gid] = prev_sum[lid];
}

__kernel void partial_copy(__global double *input, __global double *output, int input_size, int output_size) {
    uint gid = get_global_id(0);

    uint block_size = get_local_size(0);

    uint ind = gid / block_size + 1;

    if (gid < input_size && ind < output_size && 1 + gid == ind * block_size)
        output[ind] = input[gid];
}

__kernel void block_add(__global double *partial_input, __global double *input, __global double *output, int size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    if (gid < size)
        output[gid] = input[gid] + partial_input[gid / block_size];
}
