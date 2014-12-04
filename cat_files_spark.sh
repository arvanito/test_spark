#!/bin/bash

# the path of the folder is the first argument
d=$1
cd $d
echo "Concatenating files of directory $d"

# files inside this directory that contain the actual data
FILES=(`ls | grep part`)
TOTAL_FILES=${#FILES[@]}

# check if we have created modified files
# If not, modify them, if yes, jump to the concatenation
MOD_FILES=(`ls | grep mod`)
TOTAL_MOD_FILES=${#MOD_FILES[@]}
if [ ${TOTAL_MOD_FILES} -eq 0 ]; then
	# loop over each file and modify it by removing the "[","]" characters
	for ((i = 0; i < ${TOTAL_FILES}; i+=1))
	do
		# patch file
		PATCH_FILE=${FILES[$i]}

		# remove the "[","]" characters	
		OUTPUT_FILE=${PATCH_FILE}_mod
		cat ${PATCH_FILE} | sed 's/\[//g' | sed 's/\]//g' | sed 's/,/ /g' > ${OUTPUT_FILE}
	done
	echo "Finished removing [], from the patch files. Now concatenating..."
	OUTPUT_FILES=(`ls | grep mod`)
	cat ${OUTPUT_FILES[@]} > $2
else
	echo "Directly concatenate the files..."
	OUTPUT_FILES=(`ls | grep mod`)
	cat ${OUTPUT_FILES[@]} > $2
fi
cd -
