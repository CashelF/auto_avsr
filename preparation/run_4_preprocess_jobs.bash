mkdir -p logs

trap "echo 'Killing all jobs'; kill 0; exit 1" SIGINT SIGTERM

for i in $(seq 0 3); do
  echo "Starting job $i / 4"
  python -m preprocess_lrs2lrs3 \
    --data-dir ../mvlrs_v1/dataset \
    --detector mediapipe \
    --landmarks-dir ../mvlrs_v1/landmarks/LRS2_landmarks \
    --root-dir ../mvlrs_v1/preprocessed_dataset \
    --subset train \
    --dataset lrs2 \
    --gpu_type cuda \
    --groups 4 \
    --job-index $i \
    > logs/preprocess_lrs2_train_$i.log 2>&1 &
done

wait
echo "All jobs done."
