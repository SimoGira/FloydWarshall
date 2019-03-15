for filename in ../data/*.mtx; do
	for ((i=1; i<=4; i++)); do
		for ((j=0; j<5; j++)); do
        	./floydwarshall $filename $i
        done;
    done;
done

#./floydwarshall "$filename" "$1"
