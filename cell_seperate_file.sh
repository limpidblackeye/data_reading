#!/bin/bash

dir_data="/Users/april/Desktop/2018Spring/cell_nucleus/data/"
dir_gray=$dir_data"stage1_train_gray"
dir_color=$dir_data"stage1_train_color"
list_gray=$dir_data"dir_of_gray.list"
list_color=$dir_data"dir_of_color.list"

for name in `cat $list_color`
do
	echo $name
	cp -r $name $dir_color
done
