#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w speexList -s wavDir"
   echo -e "\t-s speexList is path to the file containing path to the spx files to be speex-decoded"
   echo -e "\t-w wavDir is path to the directory where output wav files will be written"
   exit 1 # Exit script after printing help
}

while getopts "s:w:" opt
do
   case "$opt" in
      s ) speexList="$OPTARG" ;;
      w ) wavDir="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# some parameters are empty, print helpFunction
if [ -z "$speexList" ] || [ -z "$wavDir" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# all parameters are correct, do the job
echo "speex-decoding speex files from $speexList"
cat $speexList | while read line || [[ -n $line ]]; # $line is a wav file
#filename="hi"
do
   filename=`echo "$line" | sed -r "s/.+\/(.+)\..+/\1/"`
   echo $filename
   speexdec $line "$wavDir/$filename.wav" #speexenc input_file.wav compressed_file.spx
done
echo "decoded wav files written to $wavDir"

