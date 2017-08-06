#include <iostream>
#include "ukf.h"

UKF::UKF() {
    //TODO Auto-generated constructor stub
    Init();
}

UKF::~UKF() {
    //TODO Auto-generated destructor stub
}

void UKF::Init() {

}

/*******************************************************************************
* Programming assignment functions:
*******************************************************************************/

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

    //set state dimension
    int n_x = 5;

    //set augmented dimension
    int n_aug = 7;

    //create example sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
    Xsig_aug <<
             5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
            1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
            2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
            0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
            0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
            0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
            0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

    double delta_t = 0.1; //time diff in sec
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

    //predict sigma points
    //avoid division by zero
    //write predicted sigma points into right column
    double px = 0.0;
    double py = 0.0;
    double v = 0.0;
    double sai = 0.0;
    double saidot = 0.0;
    double va = 0.0;
    double vsaidotdot = 0.0;

    VectorXd calCol = VectorXd(n_x);
    VectorXd xk = VectorXd(n_x);
    VectorXd eq1 = VectorXd(n_x);
    VectorXd eq2 = VectorXd(n_x);

    for (int column = 0; column < 2 * n_aug + 1; column++)
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


/*******************************************************************************
 * Student part end
 ******************************************************************************/

    //print result
    std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

    //write result
    *Xsig_out = Xsig_pred;

}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

    //set state dimension
    int n_x = 5;

    //set augmented dimension
    int n_aug = 7;

    //Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a = 0.2;

    //Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd = 0.2;

    //define spreading parameter
    double lambda = 3 - n_aug;

    //set example state
    VectorXd x = VectorXd(n_x);
    x <<   5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;

    //create example covariance matrix
    MatrixXd P = MatrixXd(n_x, n_x);
    P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

/*******************************************************************************
 * Student part begin
 ******************************************************************************/

    //create augmented mean state
    //create augmented covariance matrix
    //create square root matrix
    //create augmented sigma points

    MatrixXd Q = MatrixXd(2, 2);

    Q << std_a * std_a, 0,
            0, std_yawdd * std_yawdd;

    x_aug.head(n_x) << x;

    x_aug.tail(n_aug - n_x) << 0,0;

    P_aug.Zero(7,7);
    P_aug.topLeftCorner(n_x, n_x) = P;
    P_aug.bottomRightCorner(2,2) = Q;

    MatrixXd A = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    VectorXd cal = VectorXd(n_aug);

    for (int i = 0; i < n_aug ; i++)
    {
        cal << (sqrt(lambda + n_aug)) * A.col(i);

        Xsig_aug.col(i + 1) = x_aug + cal;
        Xsig_aug.col(i + 1 + n_aug) = x_aug - cal;

    }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

    //print result
    std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

    //write result
    *Xsig_out = Xsig_aug;

/* expected result:
   Xsig_aug =
  5.7441  5.85768   5.7441   5.7441   5.7441   5.7441   5.7441   5.7441  5.63052   5.7441   5.7441   5.7441   5.7441   5.7441   5.7441
    1.38  1.34566  1.52806     1.38     1.38     1.38     1.38     1.38  1.41434  1.23194     1.38     1.38     1.38     1.38     1.38
  2.2049  2.28414  2.24557  2.29582   2.2049   2.2049   2.2049   2.2049  2.12566  2.16423  2.11398   2.2049   2.2049   2.2049   2.2049
  0.5015  0.44339 0.631886 0.516923 0.595227   0.5015   0.5015   0.5015  0.55961 0.371114 0.486077 0.407773   0.5015   0.5015   0.5015
  0.3528 0.299973 0.462123 0.376339  0.48417 0.418721   0.3528   0.3528 0.405627 0.243477 0.329261  0.22143 0.286879   0.3528   0.3528
       0        0        0        0        0        0  0.34641        0        0        0        0        0        0 -0.34641        0
       0        0        0        0        0        0        0  0.34641        0        0        0        0        0        0 -0.34641
*/

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out)
{

    //set state dimension
    int n_x = 5;

    //define spreading parameter
    double lambda = 3 - n_x;

    //set example state
    VectorXd x = VectorXd(n_x);
    x <<   5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;

    //set example covariance matrix
    MatrixXd P = MatrixXd(n_x, n_x);
    P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

    //calculate square root of P
    MatrixXd A = P.llt().matrixL();

/*******************************************************************************
 * Student part begin
 ******************************************************************************/


    Xsig.col(0) = x;
    VectorXd cal = VectorXd(n_x);

    for (int i = 0; i < n_x ; i++)
    {
        cal << (sqrt(lambda + n_x)) * A.col(i);

        Xsig.col(i + 1) = x + cal;
        Xsig.col(i + 1 + n_x) = x - cal;

    }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

    //print result
    std::cout << "Xsig = " << std::endl << Xsig << std::endl;

    //write result
    *Xsig_out = Xsig;

/* expected result:
   Xsig =
    5.7441  5.85768   5.7441   5.7441   5.7441   5.7441  5.63052   5.7441   5.7441   5.7441   5.7441
      1.38  1.34566  1.52806     1.38     1.38     1.38  1.41434  1.23194     1.38     1.38     1.38
    2.2049  2.28414  2.24557  2.29582   2.2049   2.2049  2.12566  2.16423  2.11398   2.2049   2.2049
    0.5015  0.44339 0.631886 0.516923 0.595227   0.5015  0.55961 0.371114 0.486077 0.407773   0.5015
    0.3528 0.299973 0.462123 0.376339  0.48417 0.418721 0.405627 0.243477 0.329261  0.22143 0.286879
*/

}


//#include "ukf.h"
//#include "Eigen/Dense"
//#include <iostream>
//
//using namespace std;
//using Eigen::MatrixXd;
//using Eigen::VectorXd;
//using std::vector;
//
//*
// * Initializes Unscented Kalman filter
//
//
//UKF::UKF() {
//  // if this is false, laser measurements will be ignored (except during init)
//  use_laser_ = true;
//
//  // if this is false, radar measurements will be ignored (except during init)
//  use_radar_ = true;
//
//  // initial state vector
//  x_ = VectorXd(5);
//
//  // initial covariance matrix
//  P_ = MatrixXd(5, 5);
//
//  // Process noise standard deviation longitudinal acceleration in m/s^2
//  std_a_ = 30;
//
//  // Process noise standard deviation yaw acceleration in rad/s^2
//  std_yawdd_ = 30;
//
//  // Laser measurement noise standard deviation position1 in m
//  std_laspx_ = 0.15;
//
//  // Laser measurement noise standard deviation position2 in m
//  std_laspy_ = 0.15;
//
//  // Radar measurement noise standard deviation radius in m
//  std_radr_ = 0.3;
//
//  // Radar measurement noise standard deviation angle in rad
//  std_radphi_ = 0.03;
//
//  // Radar measurement noise standard deviation radius change in m/s
//  std_radrd_ = 0.3;
//
//*
//  TODO:
//
//  Complete the initialization. See ukf.h for other member properties.
//
//  Hint: one or more values initialized above might be wildly off...
//
//
//}
//
//UKF::~UKF() {}
//
//*
// * @param {MeasurementPackage} meas_package The latest measurement data of
// * either radar or laser.
//
//
//void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
//*
//  TODO:
//
//  Complete this function! Make sure you switch between lidar and radar
//  measurements.
//
//
//}
//
//*
// * Predicts sigma points, the state, and the state covariance matrix.
// * @param {double} delta_t the change in time (in seconds) between the last
// * measurement and this one.
//
//
//void UKF::Prediction(double delta_t) {
//*
//  TODO:
//
//  Complete this function! Estimate the object's location. Modify the state
//  vector, x_. Predict sigma points, the state, and the state covariance matrix.
//
//
//}
//
//*
// * Updates the state and the state covariance matrix using a laser measurement.
// * @param {MeasurementPackage} meas_package
//
//
//void UKF::UpdateLidar(MeasurementPackage meas_package) {
//*
//  TODO:
//
//  Complete this function! Use lidar data to update the belief about the object's
//  position. Modify the state vector, x_, and covariance, P_.
//
//  You'll also need to calculate the lidar NIS.
//
//
//}
//
//*
// * Updates the state and the state covariance matrix using a radar measurement.
// * @param {MeasurementPackage} meas_package
//
//
//void UKF::UpdateRadar(MeasurementPackage meas_package) {
//*
//  TODO:
//
//  Complete this function! Use radar data to update the belief about the object's
//  position. Modify the state vector, x_, and covariance, P_.
//
//  You'll also need to calculate the radar NIS.
//
//
//}
