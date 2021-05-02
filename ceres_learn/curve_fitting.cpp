#include <iostream>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include<chrono>
#include <math.h>
#include <algorithm>
#include <fstream>
using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
//使用define来定义一个宏
//生成a,b中的随机数
#define random(a,b) (rand()%(b-a+1)+a)
//x表示待优化参数块，residual表示残差块
//通过非线性优化找到函数梯度下降的自变量的取值。
struct CostFunctor {
    CostFunctor(double x,double y):x_(x),y_(y){
    }
  template <typename T>
  bool operator()(const T* const  m,const T* const  c,T* residual) const {
    //residual[0] = exp(m[0]*T(x_)+c[0])-T(y_);
    residual[0] = (ceres::exp(m[0]*T(x_)+c[0]))-T(y_);
    return true;
  }
private:
    const double x_,y_;
};

int main(int argc, char** argv) {
   std::ofstream openfile;
   openfile.open("/home/dongfu/projects/ceres_learn/data.txt");
    const double data[] = {
  0.000000e+00, 1.133898e+00,
  7.500000e-02, 1.334902e+00,
  1.500000e-01, 1.213546e+00,
  2.250000e-01, 1.252016e+00,
  3.000000e-01, 1.392265e+00,
  3.750000e-01, 1.314458e+00,
  4.500000e-01, 1.472541e+00,
  5.250000e-01, 1.536218e+00,
  6.000000e-01, 1.355679e+00,
  6.750000e-01, 1.463566e+00,
  7.500000e-01, 1.490201e+00,
  8.250000e-01, 1.658699e+00,
  9.000000e-01, 1.067574e+00,
  9.750000e-01, 1.464629e+00,
  1.050000e+00, 1.402653e+00,
  1.125000e+00, 1.713141e+00,
  1.200000e+00, 1.527021e+00,
  1.275000e+00, 1.702632e+00,
  1.350000e+00, 1.423899e+00,
  1.425000e+00, 1.543078e+00,
  1.500000e+00, 1.664015e+00,
  1.575000e+00, 1.732484e+00,
  1.650000e+00, 1.543296e+00,
  1.725000e+00, 1.959523e+00,
  1.800000e+00, 1.685132e+00,
  1.875000e+00, 1.951791e+00,
  1.950000e+00, 2.095346e+00,
  2.025000e+00, 2.361460e+00,
  2.100000e+00, 2.169119e+00,
  2.175000e+00, 2.061745e+00,
  2.250000e+00, 2.178641e+00,
  2.325000e+00, 2.104346e+00,
  2.400000e+00, 2.584470e+00,
  2.475000e+00, 1.914158e+00,
  2.550000e+00, 2.368375e+00,
  2.625000e+00, 2.686125e+00,
  2.700000e+00, 2.712395e+00,
  2.775000e+00, 2.499511e+00,
  2.850000e+00, 2.558897e+00,
  2.925000e+00, 2.309154e+00,
  3.000000e+00, 2.869503e+00,
  3.075000e+00, 3.116645e+00,
  3.150000e+00, 3.094907e+00,
  3.225000e+00, 2.471759e+00,
  3.300000e+00, 3.017131e+00,
  3.375000e+00, 3.232381e+00,
  3.450000e+00, 2.944596e+00,
  3.525000e+00, 3.385343e+00,
  3.600000e+00, 3.199826e+00,
  3.675000e+00, 3.423039e+00,
  3.750000e+00, 3.621552e+00,
  3.825000e+00, 3.559255e+00,
  3.900000e+00, 3.530713e+00,
  3.975000e+00, 3.561766e+00,
  4.050000e+00, 3.544574e+00,
  4.125000e+00, 3.867945e+00,
  4.200000e+00, 4.049776e+00,
  4.275000e+00, 3.885601e+00,
  4.350000e+00, 4.110505e+00,
  4.425000e+00, 4.345320e+00,
  4.500000e+00, 4.161241e+00,
  4.575000e+00, 4.363407e+00,
  4.650000e+00, 4.161576e+00,
  4.725000e+00, 4.619728e+00,
  4.800000e+00, 4.737410e+00,
  4.875000e+00, 4.727863e+00,
  4.950000e+00, 4.669206e+00,
};

const int kNumObservations = 67;
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double m=0,c=0;
  const double initial_m = 0;
  const double initial_c=0;
 // double param[]={m,c};
  // Build the problem.
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  
  for(int i=0;i<kNumObservations;i++)
  { 
      openfile<<data[2*i]<<"           "<<data[2*i+1]<<endl;
      //定义残差块，误差类型，输入维度，输出维度
      CostFunction* cost_function =new AutoDiffCostFunction<CostFunctor, 1,1,1>(new CostFunctor(data[2*i],data[2*i+1]));
      //problem.AddResidualBlock(cost_function, nullptr,&m,&c );
      //添加残差块
      problem.AddResidualBlock(cost_function,    new ceres:: CauchyLoss(0.5),&m,&c );
 }
  
      //优化问题添加残差块，不使用核函数
  //配置求解器
  ceres::Solver::Options options;
  options.linear_solver_type=ceres::DENSE_QR;//增量方程求解
  options.minimizer_progress_to_stdout=true;//输出到cout
  ceres::Solver::Summary summary;
  //开始求解
  ceres::Solve(options,&problem,&summary);
  //输出求解过程
  cout<<summary.BriefReport()<<endl;
  //输出优化后的参数
  std::cout << "m: " << initial_m<< " -> " << m << "\n";
 std::cout << "c: " << initial_c<< " -> " << c << "\n";
 openfile.close();
  return 0;
}
