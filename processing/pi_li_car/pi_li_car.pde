void setup() {
  size(500, 300);
}

void draw() {
  float deltaTime=millis();
  deltaTime=1;
  float poi=TWO_PI/30.0;
  float offset_x=TWO_PI*0.1;
  for (int i=0; i<30; i++) {

    float pipi=0.5*(1+sin(offset_x+poi*i+TWO_PI/2000*deltaTime));
    pipi=max(0.5,pipi);
    fill(pipi*255, 0, 0);
    rect(10+(i*15), 20, 10, 25);
  }
  for (int i=0; i<30; i++) {

    float pipi=0.5*(1+sin(offset_x+poi*i+TWO_PI/2000*deltaTime));
    pipi=max(0,pipi);
    fill(pipi*255, 0, 0);
    rect(10+(i*15), 50, 10, 25);
  }
  for (int i=0; i<30; i++) {

    float xx=poi*i+TWO_PI/4000*deltaTime;
    float pipi=0.5+(sin(offset_x+xx)*cos(offset_x+xx));
    pipi=max(0,pipi);
    fill(pipi*255, 0, 0);
    rect(10+(i*15), 50+30, 10, 25);
  }
}
float valueApproach(float nowVal, float destVal, float speed) {
  return  ((nowVal * (1.0 - speed)) + (destVal * speed));
}
