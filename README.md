# SPLOT
Code for implementing SPLOT method described in https://doi.org/10.1145/3379156.3391372

Full reference: Artem V. Belopolsky. 2020. Getting more out of Area of Interest (AOI) analysis
with SPLOT. In Symposium on Eye Tracking Research and Applications
(ETRA ’20 Short Papers), June 2–5, 2020, Stuttgart, Germany. ACM, New York,
NY, USA, 4 pages. https://doi.org/10.1145/3379156.3391372

ABSTRACT
To analyze eye-tracking data the viewed image is often divided
into areas of interest (AOI). However, the temporal dynamics of
eye movements towards the AOI is often lost either in favor of
summary statistics (e.g., proportion of fixations or dwell time) or
is significantly reduced by “binning” the data and computing the
same summary statistic over each time bin. This paper introduces
SPLOT: smoothed proportion of looks over time method for analyzing
the eye movement dynamics across AOI. SPLOT comprises of a
complete workflow, from visualization of the time-course to performing
statistical analysis on it using cluster-based permutation
testing. The possibilities of SPLOT are illustrated by applying it to
an existing dataset of eye movements of radiologists diagnosing a
chest X-ray.
