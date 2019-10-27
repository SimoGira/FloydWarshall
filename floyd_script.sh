for filename in ../data/*.mtx; do
	for ((i=1; i<=4; i++)); do
        #./floydwarshall $filename $i
        echo $filename $i
    done;
done

#./floydwarshall "$filename" "$1"
