import http.requests.*;
GetRequest get;
public void setup()
{
  size(400,400);
  smooth();
  get = new GetRequest("http://192.168.120.222:8080/get?accX&accY&accZ&acc");
}
float xyza[][]=new float[4][100];
int index=0;
void draw(){
  index++;
  index=index>=100?0:index;
  background(0);
  get.send();
  String result=get.getContent();
  String[][] m = matchAll(result, "\\[(.*?)\\]");
    for (int i = 0; i < m.length; i++) {
      float f=float(m[i][1]);
      xyza[i][index]=f;
      //println(char('x'+i)+"=" +f + "");    
    }
    float scale =50;
    for (int i = 0; i < xyza.length; i++) {
      //println(xyza[i]);
      fill(i==0||i==3?255:0,i==1||i==3?255:0,i==2||i==3?255:0);
      for (int j = 0; j < xyza[0].length; j++) {
        rect(width/xyza[0].length*j,50+(i*100)+xyza[i][j]*scale,5,15);
      }
    }
}
