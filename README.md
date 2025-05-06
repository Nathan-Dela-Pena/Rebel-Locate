# Rebel-Locate
Created a UNLV-Based Geolocator

In terms of implementation, this was a group led project that involved creating our own dataset that extracted metadata from each individual image taken. All together, our collective dataset amassed over 6000+ images for each individual building on campus. 
- Utilizing EXIF and OS libraries, allowed us to iterate and extract data from each buildings directory.

The next step was to identify buildings and where they were located. In terms of my individual implementation, I used KNN to predict where a building was located. By messing around with values I found that a k-value of 3 produced the most accurate results without falling victim to over and underfitting. 
- KNN model works by extracting metadata from the whole dataset and comparing it to the inputted image's coordinates. Using building labels of the nearby buildings to help determine and confirm the inputted images coordinates.
- Shapely library was also used as a secondary check to create a polygon based on outlined csv's border for each building. If an inputted image was found inside this polygon it'd confirm the location of the inputted image.

The final step was to utilize CNN to determine the inputted image's label. Using an open repository called, "Places365" allowed us to use their trained 10+ million photos weights to help our scene classifiers.
- Implementation works by appending the KNN predicted building and soley focuses on that building's unique labels.
- Extraction layers were then frozen in order to help with compilation time and soley focus on the fully connected layer. The fc layer was important as it is the trained or deterministic layer that does the thinking.
