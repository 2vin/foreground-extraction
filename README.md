# person-extraction
Extracting person from portrait images based on face localization    

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
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/test.jpeg)

After superpixelization   
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/pixels.jpg)

After kmeans segmentation   
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/segmented.jpg)

Extracted face mask   
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/facemask.jpg)

Extracted torso mask    
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/bodymask.jpg)

Combined mask     
![alt text](https://raw.githubusercontent.com/2vin/person-extraction/master/results/person.jpg)
