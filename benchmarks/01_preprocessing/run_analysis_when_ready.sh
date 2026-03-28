#!/bin/bash
# Run analysis once structure selection completes

RESULTS_DIR="structure_selection_results"

echo "Waiting for structure selection to complete..."
echo ""

# Wait for best_structures_per_uniprot.csv to be created and stable
while true; do
    if [ -f "$RESULTS_DIR/best_structures_per_uniprot.csv" ]; then
        size1=$(wc -l < "$RESULTS_DIR/best_structures_per_uniprot.csv" 2>/dev/null || echo 0)
        sleep 5
        size2=$(wc -l < "$RESULTS_DIR/best_structures_per_uniprot.csv" 2>/dev/null || echo 0)
        
        if [ "$size1" -eq "$size2" ] && [ "$size1" -gt "0" ]; then
            echo "✓ Best structures file is stable ($size1 lines)"
            break
        fi
    fi
    sleep 10
done

echo ""
echo "Running analysis on best structures..."
conda run -n plinder python3 analyze_structure_selection.py \
    --input_csv "$RESULTS_DIR/best_structures_per_uniprot.csv" \
    --output_dir "$RESULTS_DIR" \
    --top_n 50

echo ""
echo "✓ Analysis complete!"
echo ""
echo "Results available in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/*.txt "$RESULTS_DIR"/*.png 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
