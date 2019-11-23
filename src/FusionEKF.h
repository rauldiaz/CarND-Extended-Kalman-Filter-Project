#ifndef FusionEKF_H_
#define FusionEKF_H_

#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "kalman_filter.h"
#include "measurement_package.h"
#include "tools.h"

#include <fl/model/transition/linear_transition.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include "extended_kalman_filter.hpp"

using namespace fl;

class FusionEKF {
 public:
  /**
   * Constructor.
   */
  FusionEKF();

  /**
   * Destructor.
   */
  virtual ~FusionEKF();

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Kalman Filter update and prediction math lives in here.
   */
  KalmanFilter ekf_;

  typedef Eigen::VectorXd Input;
  typedef Eigen::VectorXd Obsrv;
  typedef Eigen::MatrixXd State;
  typedef Eigen::MatrixXd Noise;
  typedef LinearTransition<State, Noise, Input> Transition;
  typedef ExtendedKalmanFilter<Transition> FilterAlgorithm;
  typedef FilterInterface<FilterAlgorithm> Filter;

 private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;

  // FL
  Filter filter_;
};

#endif // FusionEKF_H_
