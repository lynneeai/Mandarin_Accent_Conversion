#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -w waveList -s speexDir"
   echo -e "\t-w waveList is path to the file containing path to the wav files to be speex-encoded"
   echo -e "\t-s speexDir is path to the directory where output speex files will be written"
   exit 1 # Exit script after printing help
}

while getopts "w:s:" opt
do
   case "$opt" in
      w ) waveList="$OPTARG" ;;
      s ) speexDir="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# some parameters are empty, print helpFunction
if [ -z "$waveList" ] || [ -z "$speexDir" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# all parameters are correct, do the job
echo "speex-encoding wav files from $waveList"
cat $waveList | while read line || [[ -n $line ]]; # $line is a wav file
#filename="hi"
do
   filename=`echo "$line" | sed -r "s/.+\/(.+)\..+/\1/"`
   echo $filename
   speexenc $line "$speexDir/$filename.spx" #speexenc input_file.wav compressed_file.spx
done
echo "encoded speex files written to $speexDir"

