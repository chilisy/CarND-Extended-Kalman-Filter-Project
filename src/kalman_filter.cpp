#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    x_ = F_*x_;
    P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    
    UpdateKalmanGainAndState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float rho_pred = sqrt(pow(px, 2)+pow(py, 2));
    float phi_pred = atan2(py, px);
    float rho_dot_pred;
    
    // if both px and py are zero, set rho_dot prediction to zero
    float tolerance = 0.0000001;
    if (rho_pred < tolerance && rho_pred > -tolerance)
    {
        rho_dot_pred = 0;
    }
    else
    {
        rho_dot_pred = (px*vx+py*vy)/rho_pred;
    }
    
        
    VectorXd z_pred(3);
    z_pred << rho_pred, phi_pred, rho_dot_pred;
    VectorXd y = z - z_pred;
    
    // phi-diff should be a value between -pi and pi
    while (y(1) > M_PI  || y(1) < -M_PI)
    {
        if (y(1) > M_PI)
            y(1) -= 2*M_PI;
        else
            y(1) += 2*M_PI;
    }
    
    UpdateKalmanGainAndState(y);
   
}

void KalmanFilter::UpdateKalmanGainAndState(const VectorXd &y)
{
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
