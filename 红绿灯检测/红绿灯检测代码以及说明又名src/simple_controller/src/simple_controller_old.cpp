#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
//#include <sensor_msgs/Joy.h>
#include <serial/serial.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
//#include <std_msgs/String.h>
#include <termios.h>///接受键盘输入
#include <sys/poll.h>
#include <boost/thread/thread.hpp>
#include <chrono>
#include <math.h>

using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using namespace std;


serial::Serial ser;

class SimpleController
{
public:
  SimpleController();
  ~SimpleController();
  void stopRobot();

  //Odometry related
  double odo_perimeter = 53.38;//cm
  int odo_wheelbase = 60;//cm
  int odo_front_overhang = 40; //cm
  int odo_rear_overhang = 20; //cm
  double odo_speed = 0.0;//     cm/s
  double odo_steer_angular = 0.0;
  int odo_round = 0;
  double odo_motor_angular = 0.0;

  double odo_x = 0.0;
  double odo_y = 0.0;
  double odo_psi = 0.0;
  double odo_last_speed = 0.0;
  int odo_times = 0;
  void updateOdometry(int steer_angular, int motor_angular, int round);


public:
  //void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
  void cmdCallback(const geometry_msgs::Twist::ConstPtr& cmd);
  ros::NodeHandle nh_;

  int linear_, angular_;

  int rocker_updown = 4;  //右边的摇杆
  int rocker_leftright = 0;//左边的摇杆

  uint8_t up_speed = 0x40;
  uint8_t down_speed = 0xe0;
  uint8_t speed_scale[4] = {0x10, 0x20, 0x30, 0x40};
  int speed_ind = 3;

  //double l_scale_;
  //double a_scale_;
  ros::Publisher odo_pub_;
  //ros::Subscriber joy_sub_;
  ros::Subscriber cmd_sub_;

  uint8_t data_buffer[8];
  int middle_angle = 1391;///yqh: 底层将舵机的方向锁死在了800-1800之间，舵机正中间的值是1391;
  int scale_angle = 652;///yqh:4096/(2pi)

private:
/*
  function： @char2hex
  params:     @char
  return:  -1 失败 or 16进制数
  单个字节转16进制数
*/
  int char2hex(char c)
  {
      if(c >= 'a' && c <= 'f')
          return c-'a'+10;
      else if(c >= 'A' && c<='F')
          return c-'A'+10;
      else if(c >= '0' && c <= '9')
          return c-'0';
      else
          return -1;      
  }


/**
 function：hexString2byte
 params: @string 16进制字符串 @uint8_t* 16进制数组 @size 16进制字符大小
 return: 16进制字符的大小
 16进制字符串按位转成16进制字符数组，转换结果直接赋给 @uint8_t 
 
**/

  int hexString2byte(const string hex, uint8_t *bytes, int size)
  {
      int len = hex.length();
      int nbytes = (len+1)/3;
      if(nbytes > size)
          return -1;
      int n;
      for(n=0; n!=nbytes; ++n)
      {
          int lndx = n*3;
          int rndx = lndx+1;
          int lbits = char2hex(hex[lndx]);
          int rbits = char2hex(hex[rndx]);
          if(lbits == -1 || rbits == -1)
              return -1;
          bytes[n] = (lbits << 4) | rbits;
      }
      return nbytes;
  }
/**
 function：int2hexString
 params：@int
 return：@string 16进制字符串

*/
  string int2hexString(int i)
  {
      stringstream ioss;
      string s_temp;
      ioss << resetiosflags(ios::uppercase) << hex << i;
      ioss >> s_temp;
      return s_temp;
  }

/**
 fuction：calculate_check_bit
 params： @uint8_t*  串口数据包
 return： @uint8_t   校验位
 默认数据包的大小为8位
*/
  uint8_t calculate_check_bit(uint8_t* buffer)
  {//buffer[8]
      int sum = 0;
      for(int i=0; i<7; i++){
          sum += buffer[i];
      }
      string temp = int2hexString(sum);
      if(temp.length() > 2){
          temp = temp.substr(temp.size() - 2);//last two nuber
      }
      uint8_t result;
      int byte = hexString2byte(temp, &result, 1);
      return result;
  }


};


void SimpleController::stopRobot()
{
    data_buffer[0] = 0xaa;
    //set angle to middle_angle
    string t_string = int2hexString(middle_angle);
    uint8_t p1;
    uint8_t p2;
    string t_string_1 = "0"+t_string.substr(0,1);
    string t_string_2 = t_string.substr(t_string.size() - 2);
    hexString2byte(t_string_1, &p1, 1);
    hexString2byte(t_string_2, &p2, 1);
    data_buffer[1] = p1;
    data_buffer[2] = p2;
    data_buffer[3] = 0x00;//set speed to zero
    data_buffer[4] = 0x00;//set speed to zero
    data_buffer[5] = 0x00;
    data_buffer[6] = 0x00;
    data_buffer[7] = calculate_check_bit(data_buffer);
    ser.write(data_buffer,sizeof(data_buffer));
}

SimpleController::~SimpleController()
{
    stopRobot();
    usleep(10000);
    ser.close();
}

SimpleController::SimpleController():
  linear_(1),
  angular_(2)
{
    string ttyusb_port;

    ros::NodeHandle ph_nh_("~");
    ph_nh_.param<string>("ttyusb_port", ttyusb_port, string("/dev/ttyUSB0"));
    //ph_nh_.param<int>("axis_linear", linear_, linear_);
    //ph_nh_.param<int>("axis_angular", angular_, angular_);

    odo_pub_ = nh_.advertise<geometry_msgs::Twist>("odo_vel", 1);
    //joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 1, &SimpleController::joyCallback, this);
    cmd_sub_ = nh_.subscribe<geometry_msgs::Twist>("cmd_vel", 1, &SimpleController::cmdCallback, this);
    try
    {
        ser.setPort(ttyusb_port.c_str());
        ser.setBaudrate(115200);
        serial::Timeout to = serial::Timeout::simpleTimeout(1000);
        ser.setTimeout(to);
        ser.open();
    }
    catch (serial::IOException& e)
    {
        ROS_ERROR_STREAM(string("Unable to open port ")+ttyusb_port);
    }

    if(ser.isOpen())
    {
        ROS_INFO_STREAM(string("Serial Port initialized")+ttyusb_port);
        stopRobot();
    }
    else
        ROS_INFO_STREAM("Serial Port Error!!!");
}

/*
void SimpleController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    //geometry_msgs::Twist twist; //测试手柄的，控制小乌龟
    //twist.angular.z = 2*joy->axes[angular_];
    //twist.linear.x = 2*joy->axes[linear_];
    //odo_pub_.publish(twist);

    int temp;
    temp = middle_angle - scale_angle*joy->axes[rocker_leftright];///yqh: 摇杆最左是1，最右是-1
    string t_string = int2hexString(temp);
    uint8_t p1;
    uint8_t p2;
    string t_string_1 = "0"+t_string.substr(0,1);
    string t_string_2 = t_string.substr(t_string.size() - 2);
    hexString2byte(t_string_1, &p1, 1);
    hexString2byte(t_string_2, &p2, 1);
    data_buffer[1] = p1;
    data_buffer[2] = p2;

    int max = 16384;//电机的扭矩电流 -16384~0~16384 对应 -20A~0~20A
    int speed = 0;
    int speed_l, speed_h;
    speed = max * joy->axes[rocker_updown];
    speed_l = speed & 0x000000ff;
    speed_h = (speed & 0x0000ff00) >> 8;
    data_buffer[3] = speed_h;
    data_buffer[4] = speed_l;

   data_buffer[7] = calculate_check_bit(data_buffer);
   ser.write(data_buffer,sizeof(data_buffer));
}*/

void SimpleController::cmdCallback(const geometry_msgs::Twist::ConstPtr & cmd){
    if(cmd->linear.x == 0){
        stopRobot();
        return;
    }
    
    int temp;
    temp = middle_angle - scale_angle*cmd->angular.z;
    if(temp > 1800) temp = 1800;
    if(temp < 800) temp = 800;
    string t_string = int2hexString(temp);
    uint8_t p1;
    uint8_t p2;
    string t_string_1 = "0"+t_string.substr(0,1);
    string t_string_2 = t_string.substr(t_string.size() - 2);
    hexString2byte(t_string_1, &p1, 1);
    hexString2byte(t_string_2, &p2, 1);
    data_buffer[1] = p1;
    data_buffer[2] = p2;

    int ratio = 16384/(15.63*odo_perimeter);//电机的扭矩电流 -16384~0~16384 对应 -20A~0~20A
    int speed = 0;
    int speed_l, speed_h;
    speed = ratio * cmd->linear.x;
    speed_l = speed & 0x000000ff;
    speed_h = (speed & 0x0000ff00) >> 8;
    data_buffer[3] = speed_h;
    data_buffer[4] = speed_l;

    data_buffer[7] = calculate_check_bit(data_buffer);
    ser.write(data_buffer,sizeof(data_buffer));
}

/**
    steer_angular：舵机扭转的角度
    motor_angular：轮胎旋转的角度
    round：        轮胎旋转的圈数  正转会增加，反转会减小
    speed = (△motro_angular/180+△round)*后轮周长
    
*/

void SimpleController::updateOdometry(int steer_angular, int motor_angular, int round){

    double differ = middle_angle-steer_angular;
    //因为左右两边的扭矩和角度对应不同，这里简单的做了个归一化处理
    odo_steer_angular = (differ>0 ? (3.14/180*(differ/500)*27.5) : (3.14/180*(differ/350)*27.5));
    odo_speed = ((motor_angular-odo_motor_angular)/180+(round - odo_round))*odo_perimeter;
    odo_speed = odo_speed<2 ? odo_speed : 0;  //限幅，有时候会读到一个很大的跳变

    //test
    geometry_msgs::Twist twist; //测试里程计  因为模型不同，这个测不出来  我用的小乌龟测
    twist.angular.z = odo_steer_angular;
    twist.linear.x = odo_speed;
    odo_pub_.publish(twist);
    //~test

    /**
     * @brief odometry process
     * β=tan−1(lr/(lf+lr)*tan(δf))  β:滑移角  lr: 后悬长度  lf:前悬长度 δf: 前轮偏角，我这里就用的舵机转角
     * ψt+1=ψt+vt/(lr+lf)*sin(β)×dt    ψt: 航向角，车辆实际角度
     * y(t+1)=yt+vt*sin(ψt+β)×dt
     * x(t+1)=xt+vtcos(ψt+β)×dt
     *
     * 
     */
    double beta = atan((double)2/3*tan(odo_steer_angular));
    odo_psi = odo_psi + (odo_speed)*cos(beta)/60.0 * tan(odo_steer_angular);
    double angular = fmod(odo_psi,3.14);
   
    //这一段是因为串口读取速度的原因，会造成前几帧都有残余数据，从而造成偏差，舍弃前10帧数据就好了
    if(odo_times > 10){
        
        double delta_speed = (odo_last_speed - odo_speed) > 0 ? (odo_last_speed - odo_speed) : (odo_speed - odo_last_speed);
        delta_speed = delta_speed < 1 ? delta_speed : 0;
        
        odo_x = odo_x + abs(odo_speed)*cos(odo_psi+beta)/100;
        odo_y = odo_y + abs(odo_speed)*sin(odo_psi + beta)/100;
       
    }else{
        odo_times++;
    }


    //~odometry process


    odo_motor_angular = motor_angular;
    odo_round = round;
    odo_last_speed = odo_speed;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_controller");
  SimpleController simple_controller;
  ros::Rate loop_rate(100);
  high_resolution_clock::time_point beginTime;
  high_resolution_clock::time_point endTime;
  int isFirst = 1;
  while(ros::ok()){

      if(ser.available()){
            ROS_INFO_STREAM("Reading to read!");
            string result;
            result = ser.readline(ser.available());
            //ROS_INFO_STREAM("Read: " << result);
            //string str = "P:3143=cnt:12580=cnt1:88779=";
            //sub[0] 里的是舵机的角度
            //sub[1] 里的是电机的角度
            //sub[2] 里的是电机旋转的圈数
            string sub[3];
            int k=0;
            bool sub_flag = false;
            beginTime = high_resolution_clock::now();

            for(int i=0; i<result.length(); i++){
                if(result[i] == ':'){
                    sub_flag = true;

                }
                if(result[i] == '='){
                    sub_flag = false;
                    k++;

                }
                if(sub_flag && result[i] != '=' && result[i] != ':'){
                    sub[k] += result[i];
                }
            }

            milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(beginTime - endTime);
            //if(endTime != NULL)
            int t = int(timeInterval.count());
            ROS_INFO_STREAM("MS: " << t);
            endTime = beginTime;

            //odometry process
            int odometry_steer = atoi(sub[0].c_str());
            int odometry_motor = atoi(sub[1].c_str());
            int odometry_round = atoi(sub[2].c_str());
            //ROS_INFO_STREAM("first: " << isFirst);
            if(!isFirst){
                simple_controller.updateOdometry(odometry_steer,odometry_motor,odometry_round);
            }
            isFirst = 0;



      }
  //ros::spin();
//  t.interrupt();
//  t.join();
//tcsetattr(kfd, TCSANOW, &cooked);

      ros::spinOnce();
      loop_rate.sleep();
  }
}


















