# Student Interaction Facial/Gesture Tracking

Using the opencv and mediapipe libaries, can be found here:
https://google.github.io/mediapipe/solutions/hands.html#max_num_hands
https://github.com/opencv/opencv/tree/master/data/haarcascades
The mediapipe is for hands, the opencv cascade is for face and eyes

The script will scan each frame of footage from a given video capture, then isolate these things:
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

