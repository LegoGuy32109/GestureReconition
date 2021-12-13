# Student Interaction Facial/Gesture Tracking

Using the opencv and mediapipe libaries, can be found here:

[Mediapipe hands pipeline][https://google.github.io/mediapipe/solutions/hands.html]

[OpenCv Haarcascades][https://github.com/opencv/opencv/tree/master/data/haarcascades]

The mediapipe is for hands, the opencv cascade is for face and eyes

The `footageAnalyze.py` script will scan each frame of footage from a given video capture, then isolate these things:
- A given rectangle where a face is detected
- A box where detected hands are (make sure to have max_hands=8)
- inside of face rectangle, eyes are attemplted to be detected 
  
It will then display those detected features to each frame
- Face rectangle in green
- Eye rectangles in red
- Hand palm area with purple dot

The footage should be filming one side of the table of at least 4 students conversing with each other to solve problems
The camera angle should include the faces of the opposing students, and try to capture the hands of all students
Example in pic below
Two cameras should be recording the table on each side, run the script for each video file


How the script determines what a hand is and what a face is is through a cascade, the result of training an algorithm to determine a face exists in an image is through training that algorithm with machine learning to discover aspects of the face. Opencv and mediapipe have done the hard work for me, I can just plug in those cascades, that can identify human features from a specific frame, for example:

### Facial Cascade
The face cascade needs a black and white image to determine the likeness of the face, so I convert the BGR footage to grayscale then plug the frame into the cascade to determine where, if any, the faces are.
### Hand Cascade
The hand cascade needs a RGB frame to determine where the hands are, it colorizes different objects uniformly in the scene so it's easier to determine solid shapes. So I use an BGRtoRGB call to convert the frame in the right format, then passes it in the hands object to process it with their cascade for hands and determine where the landmarks of the hands are. I then loop through the landmarks, it tells me where certain id points are like finger segments or the base of the hands, and if it is the base of the hand I display a purple dot to show where the computer thinks the hand is

## Proper Recording Setup
For the program to work admirably, and data captured to be useful, record students sitting across from each other solving collaberative problems (without masks). There should be two cameras, each capturing a similar picture. 2 Students looking directly at the other two students. An even group of students for each side works best. Have the cameras at an angle so that they can capture the eyes of the students, but not blocked by the back of the heads of the students in front of the camera.

### Examples
![image](https://user-images.githubusercontent.com/37216503/145740429-74e7420e-1d32-49f7-8411-cd7124f9ba8c.png)
Ideal footage will be able to locate eyes inside faces, illustrated by a red box. Here only one eye is detected though both eyes are visible

![image](https://user-images.githubusercontent.com/37216503/145740655-1f9d830a-0a68-41a5-a465-fd6b6a37e90f.png)
Hands may be covered or too close to other hands that only one hand is detected. 

![image](https://user-images.githubusercontent.com/37216503/145740827-a94fb2aa-97c3-4a04-9432-cd30cd6befaa.png)
A good angle for footage will involve capture of the opposite students hands. It's difficult to find the spot that doesn't have their heads block to much of the table, but still can show their hands reaching and pointing across the table. The software will identify hands for opposite students if it can.

![image](https://user-images.githubusercontent.com/37216503/145741353-b844b168-ee88-40f8-b9cd-485814714445.png)
Here you can see some movement in the hands, and another ideal eye capture scenario. The program is not scanning for gestures off of a static image, they keep track of the landmarks in the previous frames to determine motion and predict what the next spot will be. Duplicate features, or two heads/hands being inferred in the same spot is kept to a minimum as the program gets a better understanding of the position and accurate number of hands in the scene. 

![image](https://user-images.githubusercontent.com/37216503/145741652-e1368b6d-daf1-4ea2-9821-03616127c827.png)

Here's an example of footage from the other angle, I substitute as the other student as there are only three of us for this testing phase. Tracking motion of my hand, and identification of the other student's face. Students should not be holding anything in their hands, as it can't identify it is a hand in this example. So pencils or pens should not be incorperated when testing students. More open concept problems with them pointing at a board on the table instead of drawing things, keep the discussion verbal primarily as it's much tougher to identify collaberation when they scribbles stuff down. Also tests their communication rather than their drawing skills. 
