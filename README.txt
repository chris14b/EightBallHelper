README.txt file created for Eight Ball Helper on 26/10

Requirements:
- Python 3.7
- opencv-python 3.4.3
- numpy 1.15

To run the main program, use
$ python3 main.py <file name>
for example,
$ python3 main.py 8Ball.m4v

If the argument is a video (recommended), the video will begin to play. Once all balls have come to a stop, press either
"1" or "2" to calculate the best shot to take. "1" calculates the best shot for solids; "2" calculates the best shot for
stripes. You will see that the white ball is marked with a white circle, the eight ball with a grey circle, the solid
balls with red circles, and the striped balls with a yellow circle. The best shot, if possible, is also shown. Note
that at the very start of the game there will be no best shot, because it's near impossible to deliberately get a ball
in a pocket on the first shot. Press any key to continue playing the video. You can continue to see shots by pressing
"1" or "2". Press "ESC" to stop playing the video.

If the argument is an image, the image will be displayed with all detected features. It is assumed that the current
player wants to hit a "solids" ball.