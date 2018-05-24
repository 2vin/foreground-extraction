# foreground-extraction
Extracting foreground from portrait images based on face localization    

This code is does not produce very good results on complex backgrounds, but it contains useful code for super-pixelization, kmeans segmentation, face detection & grabcut segmentation.    

# Compile
Run the following command in terminal:    
`./compile.sh`

# Usage
./main <image_filename> <number_of_superpixels> <scale\*100> <segmentation_clusters>     
(where, <scale\*100> means the scale factor to resize image multiplied by 100)    

# Example
./main ./results/test.jpeg 100 50 5
(Note: Substitute "/home/2vin/myphoto.jpg" by path of an existing portrait image)     

# Results

Test image    
![alt text](https://github.com/2vin/color-quantize/blob/master/data/test.jpg)
