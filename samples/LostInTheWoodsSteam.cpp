#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace std;
using namespace steam;
using namespace steam::traj;

class DatasetLoader {
 private:
  // Store variable name to string (you can later convert to float, etc.)
  unordered_map<string, string> variables;

 public:
  // singlve var definitions
  double b_var;
  double d;
  double om_var;
  double r_var;
  double v_var;

  // vectors
  Eigen::VectorXd om;
  Eigen::VectorXd v;
  Eigen::VectorXd t;
  Eigen::VectorXd th_true;
  Eigen::VectorXd x_true;
  Eigen::VectorXd y_true;

  // matrices
  Eigen::MatrixXd landmarks;
  Eigen::MatrixXd bearing;
  Eigen::MatrixXd range;

  // sizes
  int size = 0;
  int n_landmarks = 0;

 public:
  void loadFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Could not open file: " << filename << "\n";
      return;
    }

    std::string line, varName, varValue;
    bool readingValue = false;

    while (std::getline(infile, line)) {
      if (line.rfind("Variable: ", 0) == 0) {
        if (!varName.empty()) {
          variables[varName] = varValue;
          varValue.clear();
        }

        varName = line.substr(10);
        readingValue = true;
      } else if (readingValue) {
        if (!varValue.empty()) varValue += " ";
        varValue += line;
      }
    }

    if (!varName.empty()) {
      variables[varName] = varValue;
    }

    infile.close();

    // Parse the variables after loading
    parseNumerics();

    cout << "Loaded " << variables.size() << " variables from " << filename
         << "\n";
  }

  // Parse a matrix from a MATLAB-style string into an Eigen::MatrixXd
  Eigen::MatrixXd parseEigenMatrix(const std::string& raw) {
    std::string s = raw;

    if (!s.empty() && s.front() == '[') s.erase(0, 1);
    if (!s.empty() && s.back() == ']') s.pop_back();

    std::vector<std::vector<double>> rows;
    std::stringstream ss(s);
    std::string rowStr;

    while (std::getline(ss, rowStr, ';')) {
      std::istringstream rowStream(rowStr);
      std::vector<double> row;
      double val;
      while (rowStream >> val) {
        row.push_back(val);
      }
      if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty()) return Eigen::MatrixXd();

    size_t numRows = rows.size();
    size_t numCols = rows[0].size();
    Eigen::MatrixXd mat(numRows, numCols);

    for (size_t i = 0; i < numRows; ++i) {
      if (rows[i].size() != numCols) {
        throw std::runtime_error("Inconsistent row sizes in matrix.");
      }
      for (size_t j = 0; j < numCols; ++j) {
        mat(i, j) = rows[i][j];
      }
    }

    return mat;
  }

  Eigen::VectorXd flattenIfVector(const Eigen::MatrixXd& mat) {
    if (mat.cols() == 1) {
      // Already a column vector
      return mat.col(0);
    } else if (mat.rows() == 1) {
      // Row vector → transpose to column
      return mat.row(0).transpose();
    } else {
      throw std::runtime_error(
          "Matrix is not a vector (1 row or 1 column required)");
    }
  }

  void parseNumerics() {
    b_var = stod(variables["b_var"]);
    d = stod(variables["d"]);
    om_var = stod(variables["om_var"]);
    r_var = stod(variables["r_var"]);
    v_var = stod(variables["v_var"]);
    om = flattenIfVector(parseEigenMatrix(variables["om"]));
    v = flattenIfVector(parseEigenMatrix(variables["v"]));
    t = flattenIfVector(parseEigenMatrix(variables["t"]));
    th_true = flattenIfVector(parseEigenMatrix(variables["th_true"]));
    x_true = flattenIfVector(parseEigenMatrix(variables["x_true"]));
    y_true = flattenIfVector(parseEigenMatrix(variables["y_true"]));
    bearing = parseEigenMatrix(variables["b"]);
    range = parseEigenMatrix(variables["r"]);
    landmarks = parseEigenMatrix(variables["l"]);
    size = y_true.size();
    n_landmarks = landmarks.rows();
  }

  void checkSizes() {
    cout << "b_var: " << b_var << "\n";
    cout << "d: " << d << "\n";
    cout << "om_var: " << om_var << "\n";
    cout << "r_var: " << r_var << "\n";
    cout << "v_var: " << v_var << "\n";
    cout << "om shape: " << om.rows() << " x " << om.cols() << "\n";
    cout << "t shape: " << t.rows() << " x " << t.cols() << "\n";
    cout << "th_true shape: " << th_true.rows() << " x " << th_true.cols()
         << "\n";
    cout << "x_true shape: " << x_true.rows() << " x " << x_true.cols() << "\n";
    cout << "y_true shape: " << y_true.rows() << " x " << y_true.cols() << "\n";
    cout << "bearing shape: " << bearing.rows() << " x " << bearing.cols()
         << "\n";
    cout << "range shape: " << range.rows() << " x " << range.cols() << "\n";
    cout << "landmarks shape: " << landmarks.rows() << " x " << landmarks.cols()
         << "\n";
    cout << "traj len: " << size << endl;
  }
};

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se3::SE3StateVar::Ptr pose;
  vspace::VSpaceStateVar<6>::Ptr velocity;
};

/** \brief Function to save data to CSV */
int saveResultToFile(vector<TrajStateVar>& states, Covariance& cov_post,
                     const string& filename) {
  // open file, print header
  ofstream poses_file(filename);
  if (poses_file.is_open()) {
    poses_file << "x,y,theta,C11,C12,C13,C22,C23,C33\n";  // Header for Pose2
    // loop through states
    bool first_state = true;
    for (const auto& state : states) {
      // extract pose (NOTE: b : Body, a : Inertial)
      lgmath::se3::Transformation pose = state.pose->value();
      Eigen::Vector3d trans = pose.r_ba_ina();  //
      Eigen::Matrix3d C_ab = pose.C_ba().transpose();
      double theta = lgmath::so3::Rotation(C_ab)
                         .vec()[2];  // extract z component of vector
      Eigen::Matrix<double, 6, 6> cov;
      if (first_state) {
        cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        first_state = false;
      } else {
        cov = cov_post.query(state.pose);
      }
      poses_file << trans[0] << "," << trans[1] << "," << theta << ","
                 << cov(0, 0) << "," << cov(0, 1) << "," << cov(0, 5) << ","
                 << cov(1, 1) << "," << cov(1, 5) << "," << cov(5, 5) << "\n";
    }
    poses_file.close();
    return 1;
  } else {
    cerr << "Error opening file" << endl;
    return 0;
  }
}

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "samples/LostInTheWoods.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Load Files
  string input_file = config["files"]["input"].as<string>();
  string output_file = config["files"]["output"].as<string>();
  string gt_output_file = config["files"]["gt_out"].as<string>();
  // Load dataset
  DatasetLoader data;
  data.loadFromFile(input_file);
  data.checkSizes();
  // switches for factors/init
  bool include_prior = config["flags"]["prior"].as<bool>();
  bool include_odom = config["flags"]["odom"].as<bool>();
  bool include_wnoa = config["flags"]["wnoa"].as<bool>();
  bool include_br_meas = config["flags"]["br"].as<bool>();
  bool gt_init = config["flags"]["gt_init"].as<bool>();
  // Get inputs from param file
  double r_max = config["params"]["r_max"].as<double>();
  double del_t = config["params"]["del_t"].as<double>();
  int start = config["params"]["start"].as<int>();
  int end = config["params"]["end"].as<int>();
  // Get noise model parameters
  auto sigma_wnoa =
      Eigen::Vector3d(config["noise"]["wnoa"].as<vector<double>>().data());
  double sigma_small = config["noise"]["odom_y"].as<double>();
  double mult_bearing = config["noise"]["bearing"].as<double>();
  double mult_range = config["noise"]["range"].as<double>();
  auto sigma_br = Eigen::Vector2d(sqrt(mult_bearing * data.b_var),
                                  sqrt(mult_range * data.r_var));

  ///
  /// Setup States
  ///

  // States
  vector<TrajStateVar> states;
  // Initialization
  if (gt_init) {
    cout << "Initializing with Ground Truth" << endl;
  } else {
    cout << "Initializing with Odometry Rollout" << endl;
  }
  for (int i = start; i <= end; i++) {
    // current velocity for odometry (NOTE: Negative signs are required)
    Eigen::Matrix<double, 6, 1> vel(-data.v[i], 0.0, 0.0, 0.0, 0.0,
                                    -data.om[i]);
    // Use gt for initial pose, or if initing from gt
    if (i == start || gt_init) {
      // get pose and vel
      Eigen::Vector3d r_bi_i(data.x_true[i], data.y_true[i], 0.0);
      Eigen::Vector3d aaxis_bi(0.0, 0.0, -data.th_true[i]);
      const auto C_bi = lgmath::so3::vec2rot(aaxis_bi);
      lgmath::se3::Transformation pose(C_bi, r_bi_i);
      // push to vector of states
      TrajStateVar temp;
      temp.time = Time(data.t(i));
      temp.pose = se3::SE3StateVar::MakeShared(pose);
      temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(vel);
      states.emplace_back(temp);
    } else {  // otherwise roll out odometry
      // Get relative pose
      const Eigen::Matrix<double, 6, 1> delta =
          states.back().velocity->value() * del_t;
      lgmath::se3::Transformation T_rel(delta);
      // push to vector of states
      TrajStateVar temp;
      temp.time = Time(i * del_t);
      temp.pose =
          se3::SE3StateVar::MakeShared(T_rel * states.back().pose->value());
      temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(vel);
      states.emplace_back(temp);
    }
  }

  // define optimization
  OptimizationProblem problem;
  // Add state variables
  for (const auto& state : states) {
    problem.addStateVariable(state.pose);
    if (include_wnoa) {
      problem.addStateVariable(state.velocity);
    }
  }

  // Setup WNOA Prior
  if (include_wnoa) {
    cout << "Adding WNOA prior" << endl;
    Eigen::Matrix<double, 6, 1> Qc_diag;
    Qc_diag << sigma_wnoa[0], sigma_wnoa[1], sigma_small, sigma_small,
        sigma_small, sigma_wnoa[2];
    traj::const_vel::Interface traj(Qc_diag);
    for (const auto& state : states)
      traj.add(state.time, state.pose, state.velocity);
    traj.addPriorCostTerms(problem);
  }

  // Add odometry measurements
  if (include_odom) {
    cout << "Adding odometry measurements" << endl;
    // Setup shared noise and loss functions
    Eigen::Vector<double, 6> cov_diag;
    double sigma_small_sq = pow(sigma_small, 2);
    cov_diag << data.v_var, sigma_small_sq, sigma_small_sq, sigma_small_sq,
        sigma_small_sq, data.om_var;
    cov_diag *= pow(del_t, 2);
    const auto noise_model =
        steam::StaticNoiseModel<6>::MakeShared(cov_diag.asDiagonal());
    const auto loss_function = steam::L2LossFunc::MakeShared();
    // Add measurements
    int ind = 0;
    for (int i = start + 1; i <= end; i++) {
      ind++;
      // Odometry Measurement
      Eigen::Matrix<double, 6, 1> odom_meas;
      odom_meas << data.v[i - 1], 0.0, 0.0, 0.0, 0.0, data.om[i - 1];
      odom_meas *= -del_t;  // NOTE velocity needs to be negated here.
      const auto pose_meas =
          se3::SE3StateVar::MakeShared(se3::SE3StateVar::T(odom_meas));
      pose_meas->locked() = true;  // lock this pose
      // define relative pose
      const auto pose_rel =
          se3::compose(states[ind].pose, se3::inverse(states[ind - 1].pose));
      // define error
      const auto error_function =
          se3::tran2vec(se3::compose(se3::inverse(pose_meas), pose_rel));
      // define cost
      const auto cost_term = WeightedLeastSqCostTerm<6>::MakeShared(
          error_function, noise_model, loss_function);
      // Add cost term
      problem.addCostTerm(cost_term);
    }
  }

  // Add range-bearing measurements
  if (include_br_meas) {
    cout << "Adding bearing range measurement factors" << endl;

    // Define noise model
    Eigen::Vector2d cov_diag_br = sigma_br.array().square();
    const auto noise_model =
        steam::StaticNoiseModel<2>::MakeShared(cov_diag_br.asDiagonal());
    const auto loss_function = steam::L2LossFunc::MakeShared();

    // Define landmarks
    vector<vspace::VSpaceStateVar<4>::Ptr> landmarks(data.n_landmarks);
    for (int j = 0; j < data.n_landmarks; j++) {
      Eigen::Vector4d tmp;
      tmp << data.landmarks(j, 0), data.landmarks(j, 1), 0.0, 1.0;
      landmarks[j] = vspace::VSpaceStateVar<4>::MakeShared(tmp);
      landmarks[j]->locked() = true;  // Fix the landmarks.
    }
    // Transformation to sensor frame
    auto r_sv_v = Eigen::Vector3d(data.d, 0.0, 0.0);
    auto C_sv = Eigen::Matrix3d::Identity();
    const auto T_sv =
        se3::SE3StateVar::MakeShared(se3::SE3StateVar::T(C_sv, r_sv_v));
    T_sv->locked() = true;
    int ind = 0;  // state index
    for (int i = start; i <= end; i++) {
      for (int j = 0; j < data.n_landmarks; j++) {
        // Check if we have a valid measurement
        if ((data.range(i, j) > 0.0) && (abs(data.bearing(i, j)) > 0.0) &&
            (data.range(i, j) < r_max)) {
          // bearing-range measurement
          auto br_meas = Eigen::Vector2d(data.bearing(i, j), data.range(i, j));
          // landmark in sensor frame
          const auto T_si = compose(T_sv, states[ind].pose);
          const auto landmark_s = stereo::compose(T_si, landmarks[j]);
          const auto error_function = p2p::br2dError(landmark_s, br_meas);
          // Add cost term to problem
          const auto cost_term = WeightedLeastSqCostTerm<2>::MakeShared(
              error_function, noise_model, loss_function);
          problem.addCostTerm(cost_term);
        }
      }
      ind++;  // update state index
    }
  }

  // Lock first pose, equivalent to prior on first pose
  if (include_prior) {
    cout << "Locking First Pose" << endl;
    states[0].pose->locked() = true;
    states[0].velocity->locked() = true;
  }

  ///
  /// Setup Solver and Optimize
  ///
  unique_ptr<GaussNewtonSolver> solver;
  string solver_select = config["params"]["solver"].as<string>();
  if (solver_select == "GN") {
    steam::GaussNewtonSolver::Params params;
    params.verbose = true;
    solver = make_unique<GaussNewtonSolver>(problem, params);

  } else if (solver_select == "LM") {
    steam::LevMarqGaussNewtonSolver::Params params;
    params.verbose = true;
    solver = make_unique<LevMarqGaussNewtonSolver>(problem, params);
  } else if (solver_select == "DL") {
    steam::DoglegGaussNewtonSolver::Params params;
    params.verbose = true;
    solver = make_unique<DoglegGaussNewtonSolver>(problem, params);
  } else {
    throw runtime_error("Selected solver is not known.");
  }
  // Run optimization
  solver->optimize();
  // Get covariance
  Covariance cov_post(*solver);
  // // Get the conditioning number at the solution
  // Eigen::SparseMatrix<double> hessian_sp;
  // Eigen::VectorXd grad;
  // problem.buildGaussNewtonTerms(hessian_sp, grad);
  // Eigen::MatrixXd hessian(hessian_sp);
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esolver(hessian);
  // Eigen::VectorXd eigenvalues = esolver.eigenvalues();
  // cout << "min hess eig : " << eigenvalues[0] << endl;
  // cout << "max hess eig : " << eigenvalues[eigenvalues.size() - 1] << endl;

  // Store to file
  saveResultToFile(states, cov_post, output_file);

  return 0;
}
