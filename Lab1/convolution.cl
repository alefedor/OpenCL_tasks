__kernel void convolution(__global double *A, __global double *B, __global double *C, int n, int m) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= n || j >= n)
        return;

    double sum = 0;

    int hm = (m - 1) / 2;

    for (int k = -hm; k <= hm; k++)
        for (int l = -hm; l <= hm; l++) {
            int x = i + k;
            int y = j + l;

            if (x < 0 || x >= n || y < 0 || y >= n) continue;

            sum += A[x * n + y] * B[(k + hm) * m + (l + hm)];
        }

    C[i * n + j] = sum;
}
