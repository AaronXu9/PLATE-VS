#!/bin/bash
# Monitor the progress of structure selection

echo "Structure Selection Progress Monitor"
echo "======================================"
echo ""

# Check if jobs are running
echo "Running processes:"
ps aux | grep "select_representative_structure.py" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13 " " $14}'
echo ""

# Check log files
if [ -f "structure_selection_results/selection_run.log" ]; then
    echo "All structures analysis log (last 5 lines):"
    tail -n 5 structure_selection_results/selection_run.log
    echo ""
fi

if [ -f "structure_selection_results/best_selection_run.log" ]; then
    echo "Best-only selection log (last 5 lines):"
    tail -n 5 structure_selection_results/best_selection_run.log
    echo ""
fi

# Check output files
if [ -f "structure_selection_results/all_structures_analyzed.csv" ]; then
    lines=$(wc -l < structure_selection_results/all_structures_analyzed.csv)
    echo "All structures analyzed: $((lines - 1)) structures processed"
fi

if [ -f "structure_selection_results/best_structures_per_uniprot.csv" ]; then
    lines=$(wc -l < structure_selection_results/best_structures_per_uniprot.csv)
    echo "Best structures selected: $((lines - 1)) UniProts processed"
fi

echo ""
echo "Expected totals:"
echo "  - All structures: 26,616"
echo "  - UniProts: 830"
