i=0
for file in ./*.$2
do
    mv "$file" "image_$i.$2"
    i=$((i+1))
done

mogrify -format png ./*.$2
