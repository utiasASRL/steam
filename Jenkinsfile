pipeline {
    agent none
    stages {
        stage("cmake flow build") {
            agent { dockerfile true }
            stages {
                stage('build deps (lgmath)') {
                    steps {
                        dir('lgmath') {
                            git branch: 'master', url: 'https://github.com/utiasASRL/lgmath.git'
                            sh '''
                                mkdir build && cd build
                                cmake ..
                                cmake --build .
                            '''
                        }
                    }
                }
                stage('build') {
                    steps {
                        sh '''
                            mkdir build && cd build
                            cmake .. -Dlgmath_DIR=`pwd`/../lgmath/build
                            cmake --build .
                            cmake --install . --prefix ../install
                        '''
                    }
                }
                stage('samples') {
                    steps {
                        dir('build') {
                            sh '''
                                ./MotionDistortedP2PandCATrajPrior
                                ./SimpleBAandCATrajPrior
                                ./SimpleBAandTrajPrior
                                ./SimpleBundleAdjustment
                                ./SimpleBundleAdjustmentFullRel
                                ./SimpleBundleAdjustmentRelLand
                                ./SimpleBundleAdjustmentRelLandX
                                ./SimpleP2PandCATrajPrior
                                ./SimplePoseGraphRelax
                                ./SimpleTrajectoryPrior
                                ./SpherePoseGraphRelax
                                ./TrustRegionExample
                            '''
                        }
                    }
                }
            }
            post {
                always {
                    deleteDir()
                }
            }
        }
        stage("ros2 flow build and test") {
            agent { dockerfile { filename 'Dockerfile.ROS2' } }
            stages {
                stage('build deps (lgmath)') {
                    steps {
                        dir('lgmath') {
                            git branch: 'master', url: 'https://github.com/utiasASRL/lgmath.git'
                            sh '''
                                source /opt/ros/foxy/setup.bash
                                colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON"
                                touch COLGON_IGNORE
                            '''
                        }
                    }
                }
                stage('build') {
                    steps {
                        sh '''
                            source `pwd`/lgmath/install/setup.bash
                            colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON"
                        '''
                    }
                }
                stage('test') {
                    steps {
                        sh '''
                            source `pwd`/lgmath/install/setup.bash
                            colcon test --event-handlers console_cohesion+
                            colcon test-result
                        '''
                    }
                }
            }
            post {
                always {
                    deleteDir()
                }
            }
        }
    }
}