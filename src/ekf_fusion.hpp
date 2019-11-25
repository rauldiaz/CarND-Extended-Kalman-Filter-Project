#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"
#include "tools.h"

#include <fl/model/transition/linear_transition.hpp>
#include <fl/model/sensor/linear_gaussian_sensor.hpp>
#include <fl/filter/gaussian/gaussian_filter.hpp>
#include "extended_kalman_filter.hpp"

using namespace fl;
using namespace Eigen;
using namespace std;

class EKFFusion {
 public:
  // FL
  typedef Eigen::VectorXd Input;
  typedef Eigen::VectorXd Obsrv;
  typedef Eigen::VectorXd State;
  typedef Eigen::MatrixXd Noise;
  typedef LinearTransition<State, Noise, Input> Transition;
  typedef Transition::Density Belief;
  typedef LinearGaussianSensor<Obsrv, State> Sensor;
  typedef ExtendedKalmanFilter<Transition, Sensor> FilterAlgorithm;
  typedef FilterInterface<FilterAlgorithm> Filter;

/**
   * Constructor.
   */
  EKFFusion()
    : filter_(transition_, sensor_),
      previous_timestamp_(0),
      is_initialized_(0)
      { }

  /**
   * Destructor.
   */
  virtual ~EKFFusion() {};

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack)
  {
    /**
     * Initialization
     */

    if (!is_initialized_) {
      /**
       * TODO: Initialize the state ekf_.x_ with the first measurement.
       * TODO: Create the covariance matrix.
       * You'll need to convert radar from polar to cartesian coordinates.
       */

      // first measurement
      cout << "EKF: " << endl;
      VectorXd x(4);
      x << 1, 1, 1, 1;

      if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // TODO: Convert radar from polar to cartesian coordinates 
        //         and initialize state.
        float ro     = measurement_pack.raw_measurements_(0);
        float phi    = measurement_pack.raw_measurements_(1);
        float ro_dot = measurement_pack.raw_measurements_(2);
        x(0) = ro     * cos(phi);
        x(1) = ro     * sin(phi);      
        x(2) = ro_dot * cos(phi);
        x(3) = ro_dot * sin(phi);

      }
      else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
        // TODO: Initialize state.
        x(0) = measurement_pack.raw_measurements_(0);
        x(1) = measurement_pack.raw_measurements_(1);
      }

      previous_timestamp_ = measurement_pack.timestamp_;
      // done initializing, no need to predict or update
      is_initialized_ = true;
      belief_.mean(x);

      return;
    }

    /**
     * Prediction
     */

    /**
     * TODO: Update the state transition matrix F according to the new elapsed time.
     * Time is measured in seconds.
     * TODO: Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

    /**
     * Predict and update our belief
     */

    Belief new_belief;
    filter_.predict(belief_, Input(), new_belief);
    belief_.mean(new_belief.mean());
    belief_.covariance(new_belief.covariance());

    /**
     * Update
     */

    /**
     * TODO:
     * - Use the sensor type to perform the update step.
     * - Update the state and covariance matrices.
     */

    filter_.update(belief_, measurement_pack.raw_measurements_, new_belief);
    belief_.mean(new_belief.mean());
    belief_.covariance(new_belief.covariance());

    // print the output
    std::cout << "x_ = " << belief_.mean() << std::endl;
    std::cout << "P_ = " << belief_.covariance() << std::endl;
  };

  const State& estimation() const
  {
      return belief_.mean();
  }


 private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;

  Transition transition_;
  Sensor sensor_;  
  FilterAlgorithm filter_;
  Belief belief_;
};

