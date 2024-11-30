java.util.LinkedList<Float> history=new java.util.LinkedList<Float>();
static float dt = 0.01;
float t = 0;
float error, old_error, i, d;
float kp=0.000, ki=0.0, kd=0.00;
float output, sensorVal_pv, setParamater=300;

void setup()
{
  size(800, 600);
  textFont(createFont("Arial", 16));
}


void pidUpdate()
{
  for (int j = 0; j < (50); j++)
  {
    error = (setParamater-sensorVal_pv);
    i += error*get_dt();
    d = ((error - old_error)/dt);
    output = (getKp()*error) + (getKi()*i) + (getKd()*d);
    old_error = error;
    sensorVal_pv += (output * dt);
    sensorVal_pv=sensorVal_pv > height?height:sensorVal_pv;
    sensorVal_pv=sensorVal_pv <0?0:sensorVal_pv;
    t += dt;
  }
  history.add(sensorVal_pv);
  if(history.size()>width)
    history.removeFirst();
}
float vaValue=0;
//void valueApproach(float nowV,float toV,float speed){
//  return nowV*(1-speed)*toV
//}
void draw()
{
  background(0);
  pidUpdate();
  strokeWeight(1);
  stroke(222);
  line(0, height-setParamater, width, height-setParamater);
  for (int i = 1; i < history.size(); i++){
    stroke((255*i / width), 64*i / width, 255-(255*i / width));
    line(i, height-history.get(i), i-1, height-history.get(i-1));
  }

  text("P: " + kp*error +"("+getKp()+")", 10, 32);
  text("I: " + ki*i+"("+getKi()+")", 10, 48);
  text("D: " + kd*d+"("+getKd()+")", 10, 64);
  text("SimTime: " + round(t) + "@dt:"+dt, 10, 80);
  text("Output: " + output, 10, 96);
  text("PV: " + round(sensorVal_pv), 10, 112);
  text("SP: " + round(setParamater), 10, 128);
  setParamater = height - mouseY;
}

void mouseClicked()
{
  setParamater = height - mouseY;
}
float get_dt() {
  return Float.MIN_NORMAL+dt+0.01;
}
float getKp() {
  return kp+0.1;
}
float getKi() {
  return ki+0.005;
}
float getKd() {
  return kd+0.01;
}
