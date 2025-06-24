## MicroCalibrate

MicroCalibrate is a framework that enables survey weight calibration integrating informative diagnostics and visualizations for the user to easily identify and work through hard-to-hit targets. A dashboard allows the user to explore the reweighting process on sample data as well as supporting uploading or pasting a link to the performance log of a specific reweighting job.

To make sure that your data can be loaded and visualized by the dashboard, it must be in csv format and contain the following columns:
- epoch (int)
- target_name (str)
- target (float)
- estimate (float)
- error (float)
- abs_error (float)
- rel_abs_error (float)
- loss (float)

To access the performance dashboard see: https://microcalibrate.vercel.app/ 
