#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter*/


UKF::UKF()
{
  is_initialized_ = false;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ <<
        0.2, 0, 0, 0, 0,
        0, 0.2, 0, 0, 0,
        0, 0, 0.2, 0, 0,
        0, 0, 0, 0.2, 0,
        0, 0, 0, 0, 0.2;

  Xsig_pred_ = MatrixXd(5, 15);
  Xsig_pred_.fill(0.0);

  previous_timestamp_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(15);
  weights_.fill(0.0);

  // set weights
  weights_(0) = lambda_/(lambda_+n_aug_);

  for (int i=1; i<2*n_aug_+1; i++)
  {  //2n+1 weights
      double weight = 0.5/(n_aug_+lambda_);
      weights_(i) = weight;
  }
}

UKF::~UKF() {}

/****************************************************************************/
//                         Initialization
/****************************************************************************/
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.*/

void UKF::Init(const MeasurementPackage &measurement_pack)
{
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        float ro = measurement_pack.raw_measurements_[0];
        float phi = measurement_pack.raw_measurements_[1];
        float ro_dot = measurement_pack.raw_measurements_[2];

        /* Converting polar to cartesian format */
        /* There is not sufficient data to use ro_dot */

        x_ << ro * sin(phi), ro * cos(phi), 2, 2, 2;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
        x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 2, 2, 2;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
}

/****************************************************************************/
//                         Sensor fusion function
/****************************************************************************/

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    if(is_initialized_)
    {
        Prediction((meas_package.timestamp_ - previous_timestamp_) / 1000000.0);

        if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_))
        {
            UpdateRadar(meas_package);
            previous_timestamp_ = meas_package.timestamp_;
        }
        else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_))

        {
            UpdateLidar(meas_package);
            previous_timestamp_ = meas_package.timestamp_;
        }
    }
    else
    {
        Init(meas_package);
        previous_timestamp_ = meas_package.timestamp_;
    }

    // print the output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
}


/****************************************************************************/
//                         Prediction functions
/****************************************************************************/
/*
*
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
*/
void UKF::Prediction(double delta_t)
{
    MatrixXd Xsig_aug = MatrixXd(7, 15);
    AugmentedSigmaPoints(&Xsig_aug);

    SigmaPointPrediction(Xsig_aug, delta_t);

    PredictMeanAndCovariance();
}

/****************************************************************************/
//                         Prediction process functions
/****************************************************************************/
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out)
{
    //creating augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //creating augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //creating sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    MatrixXd Q = MatrixXd(2, 2);

    Q << std_a_ * std_a_, 0,
            0, std_yawdd_ * std_yawdd_;

    x_aug.head(n_x_) << x_;

    x_aug.tail(n_aug_ - n_x_) << 0,0;

    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug.bottomRightCorner(2,2) = Q;

    MatrixXd A = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    VectorXd cal = VectorXd(n_aug_);

    for (int i = 0; i < n_aug_ ; i++)
    {
        cal << (sqrt(lambda_ + n_aug_)) * A.col(i);

        Xsig_aug.col(i + 1) = x_aug + cal;
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - cal;
    }

    //writing result
    *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd &Xsig_aug, double delta_t)
{
    //creating matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    double px = 0.0;
    double py = 0.0;
    double v = 0.0;
    double sai = 0.0;
    double saidot = 0.0;
    double va = 0.0;
    double vsaidotdot = 0.0;

    VectorXd calCol = VectorXd(n_x_);
    VectorXd xk = VectorXd(n_x_);
    VectorXd eq1 = VectorXd(n_x_);
    VectorXd eq2 = VectorXd(n_x_);

    for (int column = 0; column < 2 * n_aug_ + 1; column++)
    {
        px = Xsig_aug(0, column);
        py = Xsig_aug(1, column);
        v = Xsig_aug(2, column);
        sai = Xsig_aug(3, column);
        saidot = Xsig_aug(4, column);
        va = Xsig_aug(5, column);
        vsaidotdot = Xsig_aug(6, column);

        xk << px, py, v, sai, saidot;

        if(0.001 < fabs(saidot))
        {
            eq1 <<  (v/saidot) * (sin(sai + saidot * delta_t) - sin(sai)),
                    (v/saidot) * (-cos(sai + saidot * delta_t) + cos(sai)),
                    0,
                    saidot * delta_t,
                    0;
        }
        else
        {
            eq1 << v * cos(sai) * delta_t,
                    v * sin(sai) * delta_t,
                    0,
                    saidot * delta_t,
                    0;
        }

        eq2 << (0.5) * (delta_t * delta_t) * cos(sai) * va,
                (0.5) * (delta_t * delta_t) * sin(sai) * va,
                delta_t * va,
                (0.5) * (delta_t * delta_t) * vsaidotdot,
                delta_t * vsaidotdot;

        calCol = xk + eq1 + eq2;

        Xsig_pred.col(column) = calCol;
    }

    //writing result
    Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance()
{
    //creating vector for predicted state
    VectorXd x = VectorXd(n_x_);

    //creating covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    x.fill(0.0);

    //predicting state mean
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    P.fill(0.0);

    //predicting state covariance matrix
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        VectorXd x_diff = (Xsig_pred_.col(i) - x);
        //angle normalization

        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P = P + weights_(i) * x_diff * x_diff.transpose();
    }

    //write result
    x_ = x;
    P_ = P;
}

/****************************************************************************/
//                         Update Lidar function
/****************************************************************************/

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package*/
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    // Laser updates
    MatrixXd R_ = MatrixXd(2, 2);
    R_ << std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;

    MatrixXd H_ = MatrixXd(2, 5);
    H_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    VectorXd &z = meas_package.raw_measurements_;

    VectorXd y = z - (H_ * x_);
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ = x_ + (K * y);

    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);

    P_ = (I - K * H_) * P_;

    ProcessNIS(y, S, meas_package);
}

/****************************************************************************/
//                         Update radar function
/****************************************************************************/
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package*/
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    VectorXd z_pred = VectorXd(3);
    MatrixXd S_ = MatrixXd(3, 3);

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
    PredictRadarMeasurement(&z_pred, &S_, &Zsig);

    UpdateState(z_pred, S_, Zsig, meas_package);
}

/****************************************************************************/
//                         Update radar process functions
/****************************************************************************/
void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_)
{
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    //transform sigma points into measurement space

    MatrixXd R = MatrixXd(n_z,n_z);
    R << (std_radr_ * std_radr_), 0, 0,
            0, (std_radphi_ * std_radphi_), 0,
            0, 0, (std_radrd_ * std_radrd_);

    Zsig.fill(0.0);
    z_pred.fill(0.0);
    S.fill(0.0);

    double px = 0.0;
    double py = 0.0;
    double v = 0.0;
    double sai = 0.0;
    double saidot = 0.0;

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        px  = Xsig_pred_(0,i);
        py  = Xsig_pred_(1,i);
        v   = Xsig_pred_(2,i);
        sai = Xsig_pred_(3,i);
        saidot = Xsig_pred_(4,i);

        Zsig(0,i) = sqrt((px * px) + (py * py));
        Zsig(1,i) = atan2(py, px);
        Zsig(2,i) = fabs(Zsig(0,i)) > 0.001 ? ((px * cos(sai) * v + py * sin(sai) * v) / Zsig(0,i)) : 0;
    }

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        VectorXd z_diff = (Zsig.col(i) - z_pred);
        //angle normalization

        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R;

    //write result
    *z_out = z_pred;
    *S_out = S;
    *Zsig_ = Zsig;
}


void UKF::UpdateState(VectorXd z_pred, MatrixXd S, MatrixXd Zsig, MeasurementPackage &meas_package)
{
    //setting measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //creating example vector for incoming radar measurement
    VectorXd z = VectorXd(n_z);
    z <<
      meas_package.raw_measurements_[0],   //rho in m
            meas_package.raw_measurements_[1],   //phi in rad
            meas_package.raw_measurements_[2];   //rho_dot in m/s

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;

    P_ = P_ - K * S * K.transpose();

    ProcessNIS(z_diff, S, meas_package);
}

/****************************************************************************/
//                        Calculate NIS
/****************************************************************************/
void UKF::ProcessNIS(VectorXd &z_diff, MatrixXd &S, MeasurementPackage meas_package)
{
    int ret = (z_diff.transpose() * S.inverse() * z_diff);

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        cout << "NIS_radar = " << ret << endl;
        ofstream out("NisRadar.txt", ios::app);
        out << ret << endl;
        out.close();
    }
    else
    {
        cout << "NIS_lidar = " << ret << endl;
        ofstream out("NisLidar.txt", ios::app);
        out << ret << endl;
        out.close();
    }
}

