export KALDI_ROOT=`pwd`/../../kaldi-trunk
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

rm ./utils
rm ./steps
rm ./local
rm ./sid

ln -s $KALDI_ROOT/egs/voxceleb/v2/utils ./utils
ln -s $KALDI_ROOT/egs/voxceleb/v2/steps ./steps
ln -s $KALDI_ROOT/egs/voxceleb/v2/local ./local
ln -s $KALDI_ROOT/egs/voxceleb/v2/sid ./sid