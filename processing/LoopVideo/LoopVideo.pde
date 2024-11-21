
import processing.video.*;

Movie movie;
HashMap<String, String> video_files = new HashMap<String, String>();

String NOW;

void setup() {
  video_files.put("1", "/Users/xlinx/Movies/1.mp4");
  video_files.put("2", "/Users/xlinx/Movies/8.mp4");
  NOW=video_files.get("2");
  size(560, 406);
  background(0);
  // Load and play the video in a loop
  movie = new Movie(this, NOW);
  movie.loop();
}
void keyPressed() {


  if (key == '1') {
    print("LEFT");
    NOW=video_files.get("1");
    movie.stop();
    movie = new Movie(this, NOW);
    movie.loop();
  } else if (key == '2') {
    print("RIGHT");
    NOW=video_files.get("2");
    movie.stop();
    movie = new Movie(this, NOW);
    movie.loop();
  } else if (key == ' ') {
    print("pause");
    if (movie.isPaused())
      movie.play();
    else
      movie.pause();
  }
}
void movieEvent(Movie m) {
  m.read();
}

void draw() {
  background(0);
  float scale=mouseX/(float)width;
  image(movie, 0, 0, scale*width, scale*movie.height*(width/(float)movie.width));
  rectMode(CENTER);
  float scaleSize=sin(TWO_PI/9000*millis());
  rect(width/2,height/2,scaleSize*111,scaleSize*111);
}
