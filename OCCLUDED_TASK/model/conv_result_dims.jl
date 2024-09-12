### OUTPUT DIMENSION

## dimensions: given an image of size W_in x H_in
dims = (367, 398)

function W_out(W_in, K_conv, P_conv, S_conv, K_pool, S_pool)
    conv_wout = 1 + (W_in - K_conv + 2*P_conv)/S_conv
    pool_wout = 1 + (conv_wout - K_pool)/S_pool
    return pool_wout
end

function H_out(H_in, K_conv, P_conv, S_conv, K_pool, S_pool)
    conv_hout = 1 + (H_in - K_conv + 2*P_conv)/S_conv
    pool_hout = 1 + (conv_hout - K_pool)/S_pool
    return pool_hout
end

## K is the filter's dimension
println(W_out(dims[1], 3, 1, 1, 2, 2)) ## width
println(H_out(dims[2], 3, 1, 1, 2, 2)) ## height