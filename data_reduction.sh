#!/bin/bash

pwd
for file in ./Data_full/*.txt; do
	sed '2~2 d' $file |
	sed '2~2 d' |
	sed '2~2 d' |
	sed '2~2 d' > "$file 2"
done
