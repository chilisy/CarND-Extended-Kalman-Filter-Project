#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
    rmse << 0,0,0,0;
    
   
    if (estimations.size() == 0)
    {
        cout << "CalculateRMSE - Error - Estimation should not be zero!" << endl;
        return rmse;
    }
    
    if (estimations.size() != ground_truth.size())
    {
        cout << "CalculateRMSE - Error - Estimation dimension should match Ground Truth dimension" << endl;
        return rmse;
    }
    
    
    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){

        VectorXd residual = estimations[i] - ground_truth[i];
        
        residual = residual.array()*residual.array();
        rmse += residual;
        
    }
    
    //calculate the mean
    rmse /= estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    return rmse;
  
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
    Hj = MatrixXd::Zero(3, 4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    float dividend = pow(px, 2) + pow(py, 2);
    
    //check division by zero
    if (dividend == 0)
    {
        cout << "CalculateJacobian - Error - Division by zero!";
    }
    
    //compute the Jacobian matrix
    Hj = MatrixXd::Zero(3, 4);
    Hj(0, 0) = px/sqrt(dividend);
    Hj(0, 1) = py/sqrt(dividend);
    Hj(1, 0) = -py/dividend;
    Hj(1, 1) = px/dividend;
    Hj(2, 0) = py*(vx*py-vy*px)/pow(dividend, 1.5);
    Hj(2, 1) = px*(vy*px-vx*py)/pow(dividend, 1.5);
    Hj(2, 2) = px/sqrt(dividend);
    Hj(2, 3) = py/sqrt(dividend);
    
    
    return Hj;
  
}
