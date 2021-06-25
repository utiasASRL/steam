pipeline {
    agent none
    stages {
        stage("cmake flow build") {
            agent { dockerfile true }
            stages {
                stage('build') {
                    steps {
                        sh '''
                            mkdir build && cd build
                            cmake ..
                            cmake --build .
                            cmake --install . --prefix ../install
                            make doc
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
        stage("ros2 flow build and test") {
            agent { dockerfile { filename 'Dockerfile.ROS2' } }
            stages {
                stage('build') {
                    steps {
                        sh '''
                            source /opt/ros/foxy/setup.bash
                            colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON"
                            colcon build --symlink-install --cmake-args "-DUSE_AMENT=ON" --cmake-target doc
                        '''
                    }
                }
                stage('test') {
                    steps {
                        sh '''
                            source /opt/ros/foxy/setup.bash
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