#!/bin/bash

# This file is created by Lin Ai (la2734@columbia.edu), and is based on run.sh from Kaldi Voxceleb v2 recipe.

. ./cmd.sh
. ./path.sh
set -e

#author: la2734
#path and parameter definitions

# for aishell_2
partitions_dir=`pwd`/../aishell_2_partitions
aishell2_spk_test_trials=data/test/spk_trials
aishell2_acc_test_trials=data/test_acc/acc_trials
aishell2_train_name=train
aishell2_test_name=test
aishell2_train_acc_name=train_acc
aishell2_test_acc_name=test_acc

# for magicdata
magicdata_train_name=train_magicdata
magicdata_test_name=test_magicdata
magicdata_trials=data/test_magicdata/trials

# for normal training and lda-plda
normal_nnet_dir=exp/xvector_nnet_1a

# for transfer learning
trans_nnet_dir=exp/trans_nnet

dataset=magicdata
aishell2_task=acc
data_dir=`pwd`/data
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
musan_root=/home/lynnee/musan
nnet_dir=$trans_nnet_dir
train_name=$magicdata_train_name
test_name=$magicdata_test_name
trials_file=$magicdata_trials
lda_dim=32

stage=13

if [ $stage -le 0 ]; then
  #data preparation
  if [ "$dataset" = "aishell_2" ]; then
    
    #calls python_scripts/prepare_data.py to generate wav.scp, utt2spk, utt2acc
    # generate wav.scp, utt2spk
    python python_scripts/prepare_data.py --partitions_dir $partitions_dir --data_dir $data_dir --task $aishell2_task

    # generate spk2utt on train set
    utils/utt2spk_to_spk2utt.pl $data_dir/$train_name/utt2spk > $data_dir/$train_name/spk2utt

    # generate spk2utt on test set
    utils/utt2spk_to_spk2utt.pl $data_dir/$test_name/utt2spk > $data_dir/$test_name/spk2utt

    # generate spk2utt on dev set
    utils/utt2spk_to_spk2utt.pl $data_dir/dev/utt2spk > $data_dir/dev/spk2utt
  fi

  if [ "$dataset" = "magicdata" ]; then
    #author: la2734
    #calls python_scripts/prepare_magid_data.py to partite train/test sets
    #and generate wav.scp, utt2spk
    python python_scripts/prepare_magicdata_data.py
    utils/utt2spk_to_spk2utt.pl $data_dir/$train_name/utt2spk > $data_dir/$train_name/spk2utt
    utils/utt2spk_to_spk2utt.pl $data_dir/$test_name/utt2spk > $data_dir/$test_name/spk2utt
  fi
fi

if [ $stage -le 1 ]; then
	# Make MFCCs and compute the energy-based VAD for each dataset
	for name in $train_name $test_name; do
		steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
			data/${name} exp/make_mfcc $mfccdir
		utils/fix_data_dir.sh data/${name}
		sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
			data/${name} exp/make_vad $vaddir
		utils/fix_data_dir.sh data/${name}
	done
fi

# Step 2 - 3 is only used when training xvector on aishell_2.
# They are not used during transfer learning on magicdata.
# In this section, we augment the train data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/$train_name/utt2num_frames > data/$train_name/reco2dur

    if [ ! -d "RIRS_NOISES" ]; then
	    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
	    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
	    unzip rirs_noises.zip
    fi

    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    # Make a reverberated version of the train list.  Note that we don't add any
    # additive noise here.
    steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        data/$train_name data/"$train_name"_reverb
    cp data/$train_name/vad.scp data/"$train_name"_reverb/
    utils/copy_data_dir.sh --utt-suffix "-reverb" data/"$train_name"_reverb data/"$train_name"_reverb.new
    rm -rf data/"$train_name"_reverb
    mv data/"$train_name"_reverb.new data/"$train_name"_reverb

    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
        utils/data/get_utt2dur.sh data/musan_${name}
        mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done

    # Augment with musan_noise
    steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/$train_name data/"$train_name"_noise
    # Augment with musan_music
    steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/$train_name data/"$train_name"_music
    # Augment with musan_speech
    steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/$train_name data/"$train_name"_babble

    # Combine reverb, noise, music, and babble into one directory.
    utils/combine_data.sh data/"$train_name"_aug data/"$train_name"_reverb data/"$train_name"_noise data/"$train_name"_music data/"$train_name"_babble
fi

if [ $stage -le 3 ]; then
    # Take a random subset of the augmentations
    utils/subset_data_dir.sh data/"$train_name"_aug 40000 data/"$train_name"_aug_1m
    utils/fix_data_dir.sh data/"$train_name"_aug_1m

    # Make MFCCs for the augmented data.  Note that we do not compute a new
    # vad.scp file here.  Instead, we use the vad.scp from the clean version of
    # the list.
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
        data/"$train_name"_aug_1m exp/make_mfcc $mfccdir

    # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
    # double the size of the original clean list.
    utils/combine_data.sh data/"$train_name"_combined data/"$train_name"_aug_1m data/$train_name
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
    # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
    # wasteful, as it roughly doubles the amount of training data on disk.  After
    # creating training examples, this can be removed.
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 1 --cmd "$train_cmd" \
        data/$train_name data/"$train_name"_no_sil exp/"$train_name"_no_sil
    utils/fix_data_dir.sh data/"$train_name"_no_sil
fi

if [ $stage -le 5 ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 350 frames per utterance.
    min_len=350
    mv data/"$train_name"_no_sil/utt2num_frames data/"$train_name"_no_sil/utt2num_frames.bak
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/"$train_name"_no_sil/utt2num_frames.bak > data/"$train_name"_no_sil/utt2num_frames
    utils/filter_scp.pl data/"$train_name"_no_sil/utt2num_frames data/"$train_name"_no_sil/utt2spk > data/"$train_name"_no_sil/utt2spk.new
    mv data/"$train_name"_no_sil/utt2spk.new data/"$train_name"_no_sil/utt2spk
    utils/fix_data_dir.sh data/"$train_name"_no_sil

    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    min_num_utts=8
    awk '{print $1, NF-1}' data/"$train_name"_no_sil/spk2utt > data/"$train_name"_no_sil/spk2num
    awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/"$train_name"_no_sil/spk2num | utils/filter_scp.pl - data/"$train_name"_no_sil/spk2utt > data/"$train_name"_no_sil/spk2utt.new
    mv data/"$train_name"_no_sil/spk2utt.new data/"$train_name"_no_sil/spk2utt
    utils/spk2utt_to_utt2spk.pl data/"$train_name"_no_sil/spk2utt > data/"$train_name"_no_sil/utt2spk

    utils/filter_scp.pl data/"$train_name"_no_sil/utt2spk data/"$train_name"_no_sil/utt2num_frames > data/"$train_name"_no_sil/utt2num_frames.new
    mv data/"$train_name"_no_sil/utt2num_frames.new data/"$train_name"_no_sil/utt2num_frames

    # Now we're ready to create training examples.
    utils/fix_data_dir.sh data/"$train_name"_no_sil
fi

# Stages 6 through 8 are handled in run_xvector.sh
transfer_learning.sh --stage $stage --train-stage -1 \
    --data data/"$train_name"_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs

if [ $stage -le 9 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
	$nnet_dir data/$train_name \
	$nnet_dir/xvectors_"$train_name"

  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 \
	$nnet_dir data/$test_name \
	$nnet_dir/xvectors_"$test_name"
fi

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/xvectors_"$train_name"/log/compute_mean.log \
  ivector-mean scp:$nnet_dir/xvectors_"$train_name"/xvector.scp \
  $nnet_dir/xvectors_"$train_name"/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $nnet_dir/xvectors_"$train_name"/log/lda.log \
  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$train_name/xvector.scp ark:- |" \
  ark:data/$train_name/utt2spk $nnet_dir/xvectors_"$train_name"/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_"$train_name"/log/plda.log \
  ivector-compute-plda ark:data/$train_name/spk2utt \
  "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_$train_name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$train_name/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
  $nnet_dir/xvectors_"$train_name"/plda || exit 1;
fi

if [ $stage -le 11 ]; then
  $train_cmd exp/scores_"$dataset"/log/test_scoring.log \
  ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_$train_name/plda - |" \
  "ark:ivector-subtract-global-mean $nnet_dir/xvectors_$train_name/mean.vec scp:$nnet_dir/xvectors_$test_name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$train_name/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $nnet_dir/xvectors_$train_name/mean.vec scp:$nnet_dir/xvectors_$test_name/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_$train_name/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$trials_file' | cut -d\  --fields=1,2 |" exp/scores_"$test_name" || exit 1;
fi

if [ $stage -le 12 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $trials_file exp/scores_$test_name) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_$test_name $trials_file 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_$test_name $trials_file 2> /dev/null`
  echo "lda_dim: $lda_dim"
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
