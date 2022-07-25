# Read arguments

DATASET="$1" # 'voc07' or 'coco14'
OUTPUT_DIR="$2"
NUM_CYCLE="$3"
BUDGET="$4"

echo "Average and standard deviation of the performance on "$OUTPUT_DIR":"
if [[ "$DATASET" == "voc07" ]]
then 
  for cycle in $(seq 1 $NUM_CYCLE)
  do 
    echo -n "Cycle $cycle: "
    find $OUTPUT_DIR -wholename "*ver*_"$((cycle * BUDGET))"_images/inference*test*/result.txt" -not -wholename "*bbox_aug/result*" \
         -exec head -1 {} \; | awk '{m+=$2;n+=(($2)^2)}END{printf "%.1f +/- %.1f\n", 100*m/NR, 100*sqrt(n/(NR) - ((m/NR)^2) ) }'
  done
elif [[ "$DATASET" == "coco14" ]]
then
  echo "Cycle           AP              AP50"
  for cycle in $(seq 1 $NUM_CYCLE)
  do
    echo -n "$cycle" "              "
    for f in $(find $OUTPUT_DIR -wholename "*cycle"$cycle"*val*/coco_results*" -not -wholename "*bbox_aug/coco*")
    do 
      python -c "import torch; a = torch.load('"$f"'); print(a.results['bbox']['AP']*100)"
    done | awk '{m+=$1;n+=(($1)^2 )}END{printf "%.1f +/- %.1f %-3s", m/NR, sqrt(n/NR - (m/NR)^2 ), " " }'

    for f in $(find $OUTPUT_DIR -wholename "*cycle"$cycle"*val*/coco_results*" -not -wholename "*bbox_aug/coco*")
    do 
      python -c "import torch; a = torch.load('"$f"'); print(a.results['bbox']['AP50']*100)"
    done | awk '{m+=$1;n+=(($1)^2 )}END{printf "%.1f +/- %.1f\n", m/NR, sqrt(n/NR - (m/NR)^2 ) }'
  done
else 
  exit "DATASET="$DATASET" is not supported!"
  exit 1
fi