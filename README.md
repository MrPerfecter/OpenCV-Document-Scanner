📄 OpenCV Interactive Document Scanner

📌 Overview

This project is an automated, perspective-correcting document scanner built from scratch using Classical Computer Vision techniques.

Inspired by apps like CamScanner, this script takes an image of a skewed document (like a physical bill or receipt) and mathematically flattens it into a perfectly cropped, top-down 90-degree scan.

To handle varying real-world lighting conditions and background noise, I implemented a custom interactive UI with trackbars. This allows users to dynamically tune edge-detection thresholds in real-time until the document is perfectly isolated.

⚙️ The Computer Vision Pipeline

The algorithm processes the image through the following stages:

Grayscale Conversion: Reduces the color matrix to a single channel for structural analysis.

Gaussian Blur: Eliminates high-frequency background noise.

Canny Edge Detection: Computes intensity gradients to outline shapes. (Thresholds are dynamically adjustable via UI)

Morphological Dilation: Thickens edges to close any broken gaps in the document's border.

Contour Extraction: Uses cv.approxPolyDP to isolate the largest 4-point polygon, filtering out background artifacts (ignores areas < 5000 pixels).

Perspective Transformation: Calculates a warp matrix based on the 4 detected corners and applies cv.warpPerspective to flatten the image.

🚀 How to Run

1. Clone the repository

2. Install dependencies

pip install opencv-python numpy


3. Run the scanner

python scanner.py


🎮 How to Use the Interactive UI

When you run the script, a control panel will appear alongside the edge map and contour visualization.

Canny Min / Max: Slide these to adjust the edge detection sensitivity. Stop when the document border forms a solid, unbroken white loop.

Epsilon %: Adjusts the polygon approximation. If your document has slightly curved edges (like a thick book page), increase this slider slightly to force the algorithm to snap to exactly 4 corners.

The output will automatically generate once a valid 4-corner document is detected!

