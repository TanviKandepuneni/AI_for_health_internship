

Overview: In this task, I created a pipeline to process sleep breathing signals and convert them into a labeled dataset.

The goal was to:
    •    filter the signals to keep only the breathing frequency range(between 0.17 Hz and 0.4 Hz)
    •    split the signals into overlapping windows
    •    assign labels based on flow events
    •    visualize the signals
    •    save the final dataset



What the Code Does

1. Read the Data: The script reads signal files for:
    •    Flow (nasal / flow signal)
    •    Thoracic movement
    •    SPO2 (if available)
    •    Flow events (used for labeling)

The files are .txt files with timestamps and signal values.


2. Signal Filtering

Human breathing usually falls between 0.17 Hz and 0.4 Hz.

So I applied a bandpass filter (Butterworth filter using SciPy) to remove noise outside this range.

Filtering is applied to:
    •    Flow signal
    •    Thoracic signal



3. Signal Alignment

All signals are resampled to 1-second intervals so they align properly in time.


4. Windowing

The signals are split into:
    •    30-second windows
    •    with 50% overlap (step size = 15 seconds)

Each window contains:
    •    flow signal segment
    •    thoracic signal segment
    •    SPO2 signal (if available)



5. Labeling

Each window is labeled using the flow events file:
    •    If a window overlaps more than 50% with an event (like Hypopnea or Apnea), it gets that label
    •    Otherwise, it is labeled as Normal



6. Visualization (Signal Plots)

I also created a visualization script to plot the signals.

The visualization:
    •    plots flow, thoracic, and SPO2 signals over time
    •    helps check if signals look reasonable after filtering
    •    shows how signals behave during different breathing events

The output is saved as a PDF where:
    •    each page contains signal plots for a time segment
    •    signals are stacked for easier comparison

7. Saving the Dataset

The final dataset is saved in:
    •    CSV file
    •    Pickle file

Each row contains:
    •    patient ID
    •    window start and end time
    •    label
    •    flow signal values (list)
    •    thoracic signal values (list)
    •    SPO2 values (if available)

How to run: 
For create dataset: python create_dataset.py -in_dir "Data" -out_dir "Dataset"
For visualizations: python vis.py -name AP05

Output:

Dataset
    •    breathing_windows.csv
    •    breathing_windows.pkl

Visualization
    •    PDF file with signal plots for the selected patient

