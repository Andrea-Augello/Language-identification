#!/bin/bash

(
echo -e  "Input file format:\n$(head -n 20 cedict_ts.u8)";
echo -e "\n\nExtracting pinyin...";
mawk -F '[\[\],]' '{print $2}' cedict_ts.u8  > .chinese.tmp &&
echo -e "Pinyin extracted:\n$(head -n 20 .chinese.tmp)"

echo -e "\nCleaning output...";
cat .chinese.tmp | tr -d "[:blank:][:digit:]" > chinese.txt 
echo -e "Output cleaned:\n$(head -n 20 chinese.txt)"
) && 
rm .chinese.tmp
