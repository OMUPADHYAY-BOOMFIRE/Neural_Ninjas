1. Where Are Duplicate Images Stored?

All duplicate data and moved files are organized in the following locations:

Physical Duplicate Storage (duplicate/ folder):

We created the dedicated directory: c:\Users\om12u\Downloads\neural_ninja\duplicate.
All 49 redundant duplicate images have been safely moved out of Images/ into duplicate/.
The primary dataset Images/ now contains exactly 5,451 unique facial images (5,500 minus 49).

The Duplicate Mapping File (duplicate/duplicate_mapping.txt):

Inside duplicate/, we generated duplicate_mapping.txt, mapping all 39 duplicate groups.
Example excerpt from the file:
text
[Group 01 | Group ID 5006]
  RETAINED in dataset : AM578.jpg (Rating: 2.32)
  MOVED to duplicate/ : AM1509.jpg, AM1474.jpg, AM60.jpg, AM411.jpg
  All group ratings   : [2.32, 1.85, 2.35, 3.3, 2.82]
--------------------------------------------------------------------------------
[Group 02 | Group ID 1530]
  RETAINED in dataset : AM82.jpg (Rating: 2.92)
  MOVED to duplicate/ : AF886.jpg, AM1108.jpg, AM1299.jpg
  All group ratings   : [2.92, 2.25, 3.03, 2.6]
--------------------------------------------------------------------------------
[Group 29 | Group ID 3853]
  RETAINED in dataset : CM216.jpg (Rating: 2.88)
  MOVED to duplicate/ : CM439.jpg
  All group ratings   : [2.88, 3.03]

Clean Labels File (labels_clean.txt):

An updated label file labels_clean.txt was created with exactly 5,451 entries corresponding to the retained images.

| Duplicate Group | Matched Filenames | Verified Similarity |
| --- | --- | --- |
| Group 1 | `CM216.jpg` & `CM439.jpg` | Same photograph with identical hash |
| Group 2 | `CF264.jpg` & `CF294.jpg` | Same photograph with identical hash |
| Group 3 | `CF30.jpg` & `CF266.jpg` | Same photograph with identical hash |
| Group 4 | `AM980.jpg` & `AM760.jpg` | Same photograph with identical hash |
| Group 5 | `AM1913.jpg` & `AM946.jpg` | Same photograph with identical hash |
| Group 6 | `AF1001.jpg` & `AF1009.jpg` | Identical subject/lighting crop |


low_rgb (Low Compression, 
128
×
128
128×128, RGB + HSV):
Downsamples the original 
350
×
350
350×350 photo to 
128
×
128
128×128 while preserving all 3 color channels. Provides rich spatial detail and skin/hair color statistics at the expense of higher memory.

  
high_gray (High Compression, 
48
×
48
48×48, Grayscale):
Aggressively downsizes the photo to a tiny 
48
×
48
48×48 single-channel matrix. Serves as a classical ablation baseline to measure whether high resolution and color actually contribute meaningful predictive power.