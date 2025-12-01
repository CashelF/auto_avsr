set -euo pipefail

LRS2_ROOT="$(pwd)/lrs2"   # change this if needed
SEG=16                    # adjust if you preprocessed with a different --seg-duration

# 1) Move the preprocessed video segments into a single lrs2_video_seg${SEG}s tree
mkdir -p "${LRS2_ROOT}/lrs2_video_seg${SEG}s/main" "${LRS2_ROOT}/lrs2_video_seg${SEG}s/pretrain"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_train/lrs2/lrs2_video_seg${SEG}s/main/" \
  "${LRS2_ROOT}/lrs2_video_seg${SEG}s/main/"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_train/lrs2/lrs2_video_seg${SEG}s/pretrain/" \
  "${LRS2_ROOT}/lrs2_video_seg${SEG}s/pretrain/"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_val/lrs2/lrs2_video_seg${SEG}s/main/" \
  "${LRS2_ROOT}/lrs2_video_seg${SEG}s/main/"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_test/lrs2/lrs2_video_seg${SEG}s/main/" \
  "${LRS2_ROOT}/lrs2_video_seg${SEG}s/main/"

# 2) Gather labels into one labels/ directory expected by the datamodule
mkdir -p "${LRS2_ROOT}/labels"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_train/labels/lrs2_train_transcript_lengths_seg${SEG}s"* \
  "${LRS2_ROOT}/labels/"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_val/labels/lrs2_val_transcript_lengths_seg${SEG}s"* \
  "${LRS2_ROOT}/labels/"

rsync -ah --remove-source-files \
  "${LRS2_ROOT}/preprocessed_test/labels/lrs2_test_transcript_lengths_seg${SEG}s"* \
  "${LRS2_ROOT}/labels/"

# 3) (Optional) remove the now-empty per-split folders
# find "${LRS2_ROOT}" -type d -empty -name "preprocessed_*" -delete