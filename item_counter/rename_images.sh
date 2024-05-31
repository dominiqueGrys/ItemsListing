num=1
for file in *.png; do
  new=$(printf "image_%02d.png" "$num")  # Adjust the padding %02d for two digits
  mv -i -- "$file" "$new"
  num=$((num + 1))
done
