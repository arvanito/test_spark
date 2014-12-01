#!/bin/bash

# the path of the folder is the first argument
d=$1
cd $d
echo "Concatenating files of directory $d"

# files inside this directory that contain the actual data
FILES=(`ls | grep part`)
TOTAL_FILES=${#FILES[@]}

# loop over each file and modify it by removing the "[","]" characters
for ((i = 0; i < ${TOTAL_FILES}; i+=1))
do
	# patch file
	PATCH_FILE=${FILES[$i]}

	# remove the "[","]" characters	
	cat ${PATCH_FILE} | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/ /g' > ${PATCH_FILE}

done
echo "Finished removing [], from the patch files. Now concatenating..."
cat ${FILES[@]} > $2
cd -
