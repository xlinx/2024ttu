

void setup(){
  size(500,200);
}
int count=10;
void draw(){
  for(int i=0;i<count;i++){
    float offset=TWO_PI/count;
    float nowR=0.5*(offset+1+sin(TWO_PI/3000.0*millis()));
    fill(nowR*255,0,0);
    rect(50+(i*20),50,10,10);
  }
  
}
