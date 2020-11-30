#!/bin/bash

# This file is created by Lin Ai (la2734@columbia.edu), and is based on run_xvector.sh from Kaldi Voxceleb v2 recipe.

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=data/train
nnet_dir=exp/xvector_nnet_1a/
egs_dir=exp/xvector_nnet_1a/egs
src_mdl=exp/xvector_nnet_1a/final.raw

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

if [ $stage -le 6 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/voxceleb2/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 20000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 150 \
    --max-frames-per-chunk 350 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$data" $egs_dir
fi

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  echo " generating new layers";
  num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
  feat_dim=$(cat $egs_dir/info/feat_dim)

  # This chunk-size corresponds to the maximum number of frames the
  # stats layer is able to pool over.  In this script, it corresponds
  # to 100 seconds.  If the input recording is greater than 100 seconds,
  # we will compute multiple xvectors from the same recording and average
  # to produce the final xvector.
  max_chunk_size=10000

  # The smallest number of frames we're comfortable computing an xvector from.
  # Note that the hard minimum is given by the left and right context of the
  # frame-level layers.
  min_chunk_size=25
  mkdir -p $nnet_dir/configs
  cat <<EOF > $nnet_dir/configs/network.xconfig
  # new layers after tdnn6
  relu-layer name=new_layer1 input=tdnn6.affine dim=256
  relu-layer name=new_layer2 dim=128
  relu-layer name=new_layer3 dim=64
  output-layer name=output include-log-softmax=true dim=${num_targets}
EOF

  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
      --xconfig-file $nnet_dir/configs/network.xconfig \
      --config-dir $nnet_dir/configs/
  
  $train_cmd $nnet_dir/log/generate_new_layers.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=0" $src_mdl - \| \
      nnet3-init --srand=1 - $nnet_dir/configs/final.config $nnet_dir/input.raw  || exit 1;

  cp $nnet_dir/configs/final.config $nnet_dir/nnet.config

  echo "output-node name=output input=output.log-softmax" > $nnet_dir/extract.config
  echo "$max_chunk_size" > $nnet_dir/max_chunk_size
  echo "$min_chunk_size" > $nnet_dir/min_chunk_size
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -le 8 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model $nnet_dir/input.raw \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.05 \
    --trainer.optimization.final-effective-lrate=0.005 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=10 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --dir=$nnet_dir  || exit 1;
fi

exit 0;
