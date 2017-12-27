for i in {10..13};
do
    for j in {1..4};
    do python conv_add2_network.py $((i*50)) $((j*30));
    done;
done;
